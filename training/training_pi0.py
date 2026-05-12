import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Config, PI0Policy

@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    horizon: int = 16 
    fps: int = 10
    batch_size: int = 8  # Reduced for PI0 as it is a large model
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    log_freq: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    use_compile: bool = True
    num_workers: int = 4
    
    eval_freq: int = 10
    num_eval_episodes: int = 5
    save_dir: str = "checkpoints/pi0_pusht"

def rollout_and_evaluate(policy, env_id, num_episodes, device, preprocessor, action_stats):
    """Run simulation rollouts to evaluate the policy."""
    import imageio
    import numpy as np
    import gymnasium as gym
    import gym_pusht
    
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.NameNotFound:
        print(f"\n⚠️ Environment {env_id} not found. Skipping evaluation.")
        return 0.0, []

    policy.eval()
    successes = []
    best_video_frames = []
    
    with torch.no_grad():
        for ep in range(num_episodes):
            obs, info = env.reset()
            # Filter initial state to be within dataset distribution
            reset_count = 0
            while (obs[0] < 120 or obs[0] > 380 or obs[1] < 120 or obs[1] > 380) and reset_count < 100:
                obs, info = env.reset()
                reset_count += 1
                
            done = False
            if hasattr(policy, "reset"):
                policy.reset()
                
            ep_frames = []
            frame = env.render()
            if frame is not None:
                ep_frames.append(frame)
                
            step_count = 0
            while not done and step_count < 1000:
                batch = {}
                
                if "observation.image" in policy.config.input_features:
                    import torchvision.transforms as T
                    transform = T.Compose([
                        T.ToPILImage(),
                        T.Resize((224, 224)), # PI0 usually expects 224x224
                        T.ToTensor(),
                    ])
                    img_tensor = transform(frame).to(device)
                    batch["observation.image"] = img_tensor.unsqueeze(0)
                
                if "observation.state" in policy.config.input_features:
                    state_tensor = torch.from_numpy(obs[:2]).float().to(device)
                    batch["observation.state"] = state_tensor.unsqueeze(0)
                
                batch = preprocessor(batch)
                
                action = policy.select_action(batch)
                
                # Unnormalize action (PI0 uses MEAN_STD)
                mean = torch.from_numpy(action_stats['mean']).to(device)
                std = torch.from_numpy(action_stats['std']).to(device)
                unnorm_action = action * std + mean
                
                action_np = unnorm_action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                step_count += 1
                
                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame)
            
            is_success = info.get("is_success", False) or info.get("success", False) or reward > 0.9
            successes.append(1.0 if is_success else 0.0)
            
            if len(best_video_frames) == 0 or (is_success and sum(successes) == 1):
                best_video_frames = ep_frames
                
    env.close()
    policy.train()
    success_rate = sum(successes) / num_episodes
    return success_rate, best_video_frames

def main():
    cfg = TrainConfig()

    delta_timestamps = {
        "action": [t / cfg.fps for t in range(cfg.horizon)],
        "observation.image": [0],
        "observation.state": [0],
    }

    dataset = LeRobotDataset(cfg.dataset_id, delta_timestamps=delta_timestamps)

    print(f"Total frames in dataset: {len(dataset)}")
    print(f"Total episodes: {dataset.num_episodes}")

    total_episodes = dataset.num_episodes
    val_split_idx = int(total_episodes * 0.9)

    train_sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"][:val_split_idx],
        dataset.meta.episodes["dataset_to_index"][:val_split_idx],
        drop_n_last_frames=cfg.horizon,
        shuffle=True,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device.startswith("cuda") else False,
    )

    device = torch.device(cfg.device)

    print("\nInstantiating PI0 policy...")
    input_features = {}
    for key, ft in dataset.meta.features.items():
        if key.startswith("observation.image"):
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=tuple(ft["shape"]))
        elif key == "observation.state":
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(ft["shape"]))
            
    output_features = {}
    for key, ft in dataset.meta.features.items():
        if key == "action":
            output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=tuple(ft["shape"]))

    config = PI0Config(
        input_features=input_features,
        output_features=output_features,
        chunk_size=cfg.horizon,
    )

    policy = PI0Policy(config)
    policy.to(device)

    if cfg.use_compile and cfg.device.startswith("cuda"):
        print("\nCompiling policy with torch.compile()...")
        policy = torch.compile(policy)

    print("\nCreating preprocessor...")
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset.meta.stats
    )

    optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device='cuda') if cfg.use_amp and cfg.device.startswith("cuda") else None

    print(f"\nStarting training for {cfg.num_epochs} epochs on {device}...")

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_success_rate = -1.0

    policy.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = preprocessor(batch)
            
            optimizer.zero_grad(set_to_none=True)
            
            if cfg.use_amp and cfg.device.startswith("cuda"):
                pt_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=pt_dtype):
                    loss, _ = policy.forward(batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % cfg.log_freq == 0:
                print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_dataloader)
        print(f"==> Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Evaluation
        if (epoch + 1) % cfg.eval_freq == 0 or (epoch + 1) == cfg.num_epochs:
            print(f"\n--- Running Evaluation Rollouts ---")
            env_id = "gym_pusht/PushT-v0"
            success_rate, video_frames = rollout_and_evaluate(
                policy, env_id, cfg.num_eval_episodes, device, preprocessor, dataset.meta.stats["action"]
            )
            print(f"==> Epoch {epoch+1} Success Rate: {success_rate * 100:.1f}%")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_path = os.path.join(cfg.save_dir, "best_model")
                print(f"New best success rate! Saving model to {save_path}")
                # Save compiled model requires unwrapping
                model_to_save = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
                model_to_save.save_pretrained(save_path)
                
                if len(video_frames) > 0:
                    import imageio
                    video_path = os.path.join(cfg.save_dir, f"best_epoch_{epoch+1}.mp4")
                    imageio.mimsave(video_path, video_frames, fps=10)
                    print(f"Saved video to {video_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
