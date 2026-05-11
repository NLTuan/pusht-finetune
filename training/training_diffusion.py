import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConfig, DiffusionPolicy

@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    horizon: int = 16          # Replaces chunk_size in ACT
    n_obs_steps: int = 2       # Diffusion typically uses a history of observations
    n_action_steps: int = 8    # How many future actions to execute
    fps: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5
    log_freq: int = 10
    noise_scheduler_type: str = "DDPM"  # Options: "DDPM", "DDIM"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    use_compile: bool = True
    num_workers: int = 4

def main():
    cfg = TrainConfig()

    # Diffusion requires historical observations in its delta_timestamps
    delta_timestamps = {
        "action": [t / cfg.fps for t in range(cfg.horizon)],
        "observation.image": [t / cfg.fps for t in range(-cfg.n_obs_steps + 1, 1)],
        "observation.state": [t / cfg.fps for t in range(-cfg.n_obs_steps + 1, 1)],
    }

    dataset = LeRobotDataset(cfg.dataset_id, delta_timestamps=delta_timestamps)

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        drop_n_last_frames=cfg.horizon,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device.startswith("cuda") else False,
    )

    print(f"Total frames in dataset: {len(dataset)}")
    print(f"Total episodes: {dataset.num_episodes}")

    device = torch.device(cfg.device)
    if cfg.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\nInstantiating Diffusion policy with dynamic configs...")
    
    # Dynamically extract features from dataset
    input_features = {}
    for key, ft in dataset.meta.features.items():
        if key.startswith("observation.image"):
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=tuple(ft["shape"]))
        elif key == "observation.state":
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(ft["shape"]))
        elif key == "language_instruction":
            # Support language conditioning if present in the dataset
            input_features[key] = PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,))
            
    output_features = {}
    for key, ft in dataset.meta.features.items():
        if key == "action":
            output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=tuple(ft["shape"]))

    config = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        noise_scheduler_type=cfg.noise_scheduler_type,
    )

    policy = DiffusionPolicy(config)
    policy.to(device)

    if cfg.use_compile and cfg.device.startswith("cuda"):
        print("\nCompiling policy with torch.compile()...")
        policy = torch.compile(policy)

    print("\nCreating preprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset.meta.stats
    )

    optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device='cuda') if cfg.use_amp and cfg.device.startswith("cuda") else None

    print(f"\nStarting training for {cfg.num_epochs} epochs on {device}...")

    policy.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            batch = preprocessor(batch)
            
            optimizer.zero_grad(set_to_none=True)
            
            if cfg.use_amp and cfg.device.startswith("cuda"):
                pt_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=pt_dtype):
                    loss, output_dict = policy.forward(batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, output_dict = policy.forward(batch)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % cfg.log_freq == 0:
                print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    main()
