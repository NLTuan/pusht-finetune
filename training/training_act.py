import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import gymnasium as gym
from dataclasses import dataclass

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy

@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    action_chunk_size: int = 50
    fps: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5
    log_freq: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    use_compile: bool = True
    num_workers: int = 4
    
    # Eval and Logging
    eval_freq: int = 5
    num_eval_episodes: int = 5
    save_dir: str = "checkpoints/act_pusht"
    use_wandb: bool = True
    wandb_project: str = "pusht-finetune"

def rollout_and_evaluate(policy, env_id, num_episodes, device):
    """Run simulation rollouts to evaluate the policy."""
    import imageio
    import numpy as np
    
    # NOTE: You may need to import your specific env registration here (e.g., import gym_pusht)
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.NameNotFound:
        print(f"\n⚠️ Environment {env_id} not found. Skipping evaluation.")
        print("Please ensure your environment is registered with gymnasium.")
        return 0.0, []

    policy.eval()
    successes = []
    best_video_frames = []
    
    with torch.no_grad():
        for ep in range(num_eval_episodes=num_episodes):
            obs, info = env.reset()
            done = False
            
            # Clear observation history / queues in the policy
            if hasattr(policy, "reset"):
                policy.reset()
                
            ep_frames = []
            
            # We assume a max step limit if the env doesn't truncate
            step_count = 0
            while not done and step_count < 1000:
                batch = {}
                for k, v in obs.items():
                    if k in policy.config.input_features:
                        # Convert numpy to tensor, add batch dimension, move to device
                        t = torch.from_numpy(np.array(v)).float().to(device).unsqueeze(0)
                        batch[k] = t
                
                # select_action automatically handles normalization and action chunking history internally!
                action = policy.select_action(batch)
                action_np = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                step_count += 1
                
                # Render frame for video
                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame)
            
            is_success = info.get("is_success", False) or info.get("success", False) or reward > 0.9
            successes.append(1.0 if is_success else 0.0)
            
            # Save the video of the first successful rollout (or the last rollout if none succeed)
            if len(best_video_frames) == 0 or (is_success and sum(successes) == 1):
                best_video_frames = ep_frames
                
    env.close()
    policy.train()
    success_rate = sum(successes) / num_episodes
    return success_rate, best_video_frames

def main():
    cfg = TrainConfig()

    delta_timestamps = {"action": [t / cfg.fps for t in range(cfg.action_chunk_size)]}

    dataset = LeRobotDataset(cfg.dataset_id, delta_timestamps=delta_timestamps)

    print(f"Total frames in dataset: {len(dataset)}")
    print(f"Total episodes: {dataset.num_episodes}")

    # Split dataset into 90% Train / 10% Validation based on episodes
    total_episodes = dataset.num_episodes
    val_split_idx = int(total_episodes * 0.9)
    print(f"Splitting into {val_split_idx} Train episodes and {total_episodes - val_split_idx} Val episodes.")

    # Training Sampler (Episodes 0 to val_split_idx)
    train_sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"][:val_split_idx],
        dataset.meta.episodes["dataset_to_index"][:val_split_idx],
        drop_n_last_frames=cfg.action_chunk_size,
        shuffle=True,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device.startswith("cuda") else False,
    )

    # Validation Sampler (Episodes val_split_idx to end)
    val_sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"][val_split_idx:],
        dataset.meta.episodes["dataset_to_index"][val_split_idx:],
        drop_n_last_frames=cfg.action_chunk_size,
        shuffle=False,
    )

    val_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device.startswith("cuda") else False,
    )

    device = torch.device(cfg.device)
    if cfg.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\nInstantiating ACT policy with dynamic configs...")
    
    # Dynamically extract features from dataset
    input_features = {}
    for key, ft in dataset.meta.features.items():
        if key.startswith("observation.image"):
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=tuple(ft["shape"]))
        elif key == "observation.state":
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(ft["shape"]))
        elif key == "language_instruction":
            # Support language conditioning if present
            input_features[key] = PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,))
            
    output_features = {}
    for key, ft in dataset.meta.features.items():
        if key == "action":
            output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=tuple(ft["shape"]))

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=cfg.action_chunk_size,
        n_action_steps=cfg.action_chunk_size,
    )

    policy = ACTPolicy(config)
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

    if cfg.use_wandb:
        # We can pass the dataclass fields directly into wandb config
        from dataclasses import asdict
        wandb.init(project=cfg.wandb_project, config=asdict(cfg), name="act_pusht")
        print("\nWeights & Biases logging enabled.")

    print(f"\nStarting training for {cfg.num_epochs} epochs on {device}...")

    best_success_rate = -1.0
    global_step = 0

    policy.train()
    for epoch in range(cfg.num_epochs):
        total_train_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Preprocess batch (normalize)
            batch = preprocessor(batch)
            
            optimizer.zero_grad(set_to_none=True) # Faster than zero_grad()
            
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
            
            total_train_loss += loss.item()
            global_step += 1
            
            if batch_idx % cfg.log_freq == 0:
                print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                if cfg.use_wandb:
                    wandb.log({"train/step_loss": loss.item(), "global_step": global_step})
                
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"==> Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        if cfg.use_wandb:
            wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})
            
        # ==========================================
        # OFFLINE VALIDATION (Loss on val split)
        # ==========================================
        policy.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                val_batch = preprocessor(val_batch)
                
                if cfg.use_amp and cfg.device.startswith("cuda"):
                    pt_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    with torch.amp.autocast(device_type="cuda", dtype=pt_dtype):
                        v_loss, _ = policy.forward(val_batch)
                else:
                    v_loss, _ = policy.forward(val_batch)
                    
                total_val_loss += v_loss.item()
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"==> Epoch {epoch+1} Average Val Loss: {avg_val_loss:.4f}")
        if cfg.use_wandb:
            wandb.log({"eval/val_loss": avg_val_loss, "epoch": epoch})
            
        policy.train() # Set back to train mode

        # ==========================================
        # EVALUATION ROLLOUTS & CHECKPOINTING
        # ==========================================
        # Save latest model every epoch
        os.makedirs(cfg.save_dir, exist_ok=True)
        policy.save_pretrained(os.path.join(cfg.save_dir, "latest_model"))
        
        # Run online evaluation
        if (epoch + 1) % cfg.eval_freq == 0:
            print(f"\n--- Running Evaluation Rollouts for {cfg.num_eval_episodes} episodes ---")
            
            # We map dataset ID to gym env ID if needed, e.g. lerobot/pusht -> gym_pusht/PushT-v0
            env_id = "gym_pusht/PushT-v0" if "pusht" in cfg.dataset_id else cfg.dataset_id
            
            success_rate, video_frames = rollout_and_evaluate(policy, env_id, cfg.num_eval_episodes, device)
            print(f"Evaluation Success Rate: {success_rate * 100:.1f}%")
            
            if cfg.use_wandb:
                eval_metrics = {"eval/success_rate": success_rate, "epoch": epoch}
                
                if len(video_frames) > 0:
                    import numpy as np
                    # Wandb Video expects shape (time, channel, height, width)
                    vid_tensor = np.array(video_frames).transpose(0, 3, 1, 2)
                    eval_metrics["eval/rollout_video"] = wandb.Video(vid_tensor, fps=cfg.fps, format="mp4")
                    
                wandb.log(eval_metrics)
            
            # Save best model
            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                print(f"🌟 New best model! Saving to {os.path.join(cfg.save_dir, 'best_model')}")
                policy.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
                
    if cfg.use_wandb:
        wandb.finish()
    print("\nTraining complete!")

if __name__ == "__main__":
    main()