import os
import time
import torch
import torch.optim as optim
import copy
from contextlib import nullcontext
from torch.utils.data import DataLoader
import wandb
import gymnasium as gym
from dataclasses import dataclass

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.transforms import ImageTransforms, ImageTransformsConfig
from transformers import get_cosine_schedule_with_warmup

@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    horizon: int = 16          # Replaces chunk_size in ACT
    n_obs_steps: int = 2       # Diffusion typically uses a history of observations
    n_action_steps: int = 8    # How many future actions to execute
    fps: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 800
    log_freq: int = 50
    noise_scheduler_type: str = "DDPM"  # Options: "DDPM", "DDIM"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    use_compile: bool = True
    num_workers: int = 4
    
    num_inference_steps: int = 100
    eval_freq: int = 50
    num_eval_episodes: int = 10
    save_dir: str = "checkpoints/diffusion_pusht"
    use_wandb: bool = True
    wandb_project: str = "pusht-finetune-diffusion"

def rollout_and_evaluate(policy, env_id, num_episodes, device, preprocessor, postprocessor):
    """Run simulation rollouts to evaluate the policy."""
    import numpy as np
    import gym_pusht
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.NameNotFound:
        print(f"\n⚠️ Environment {env_id} not found. Skipping evaluation.")
        print("Please ensure your environment is registered with gymnasium.")
        return 0.0, 0.0, 0.0, []

    policy.eval()
    successes = []
    ever_successes = []
    max_coverages = []
    best_video_frames = []

    # Hoist constants out of the inner loop
    input_features = policy.config.input_features
    has_image = "observation.image" in input_features
    has_state = "observation.state" in input_features

    if has_image:
        import torchvision.transforms as T
        img_transform = T.Compose([T.ToPILImage(), T.Resize((96, 96)), T.ToTensor()])

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, info = env.reset()
            reset_count = 0
            while (obs[0] < 120 or obs[0] > 380 or obs[1] < 120 or obs[1] > 380) and reset_count < 100:
                obs, info = env.reset()
                reset_count += 1

            done = False
            ever_succeeded = False
            current_max_coverage = 0.0

            if hasattr(policy, "reset"):
                policy.reset()

            ep_frames = []
            frame = env.render()
            if frame is not None:
                ep_frames.append(frame)

            step_count = 0
            while not done and step_count < 1000:
                batch = {}
                if has_image:
                    batch["observation.image"] = img_transform(frame).to(device).unsqueeze(0)
                if has_state:
                    batch["observation.state"] = torch.from_numpy(obs[:2]).float().to(device).unsqueeze(0)

                batch = preprocessor(batch)
                action = policy.select_action(batch)

                # Unnormalize via LeRobot postprocessor — same pipeline as lerobot-rollout
                action_np = postprocessor(action).squeeze(0).numpy()

                obs, reward, terminated, truncated, info = env.step(action_np)
                current_max_coverage = max(current_max_coverage, reward)
                if reward > 0.9 or info.get("is_success", False) or info.get("success", False):
                    ever_succeeded = True
                done = terminated or truncated
                step_count += 1

                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame)

            is_success = info.get("is_success", False) or info.get("success", False) or reward > 0.9
            successes.append(1.0 if is_success else 0.0)
            ever_successes.append(1.0 if ever_succeeded else 0.0)
            max_coverages.append(current_max_coverage)
            if len(best_video_frames) == 0 or (is_success and sum(successes) == 1):
                best_video_frames = ep_frames

    env.close()
    policy.train()
    return sum(successes) / num_episodes, sum(ever_successes) / num_episodes, sum(max_coverages) / num_episodes, best_video_frames

def main():
    cfg = TrainConfig()

    # Diffusion requires historical observations in its delta_timestamps
    delta_timestamps = {
        "action": [t / cfg.fps for t in range(cfg.horizon)],
        "observation.image": [t / cfg.fps for t in range(-cfg.n_obs_steps + 1, 1)],
        "observation.state": [t / cfg.fps for t in range(-cfg.n_obs_steps + 1, 1)],
    }

    # Training dataset — with image augmentations
    image_transforms = ImageTransforms(ImageTransformsConfig(enable=True))
    dataset = LeRobotDataset(cfg.dataset_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms)

    # Separate validation dataset — NO augmentations for a stable, clean val loss curve
    val_dataset = LeRobotDataset(cfg.dataset_id, delta_timestamps=delta_timestamps, image_transforms=None)

    print(f"Total frames in dataset: {len(dataset)}")
    print(f"Total episodes: {dataset.num_episodes}")

    # Split dataset into 90% Train / 10% Validation based on episodes
    total_episodes = dataset.num_episodes
    val_split_idx = int(total_episodes * 0.9)
    print(f"Splitting into {val_split_idx} Train episodes and {total_episodes - val_split_idx} Val episodes.")

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
        persistent_workers=cfg.num_workers > 0,
    )

    val_sampler = EpisodeAwareSampler(
        val_dataset.meta.episodes["dataset_from_index"][val_split_idx:],
        val_dataset.meta.episodes["dataset_to_index"][val_split_idx:],
        drop_n_last_frames=cfg.horizon,
        shuffle=False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device.startswith("cuda") else False,
        persistent_workers=cfg.num_workers > 0,
    )

    device = torch.device(cfg.device)
    if cfg.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\nInstantiating Diffusion policy with dynamic configs...")
    
    input_features = {}
    for key, ft in dataset.meta.features.items():
        if key.startswith("observation.image"):
            shape = (ft["shape"][2], ft["shape"][0], ft["shape"][1])
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
        elif key == "observation.state":
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(ft["shape"]))
        elif key == "language_instruction":
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
        num_inference_steps=cfg.num_inference_steps,
        resize_shape=(96, 96),
        crop_shape=(84, 84),
        crop_is_random=True,
        use_group_norm=True,
        pretrained_backbone_weights=None,
    )

    policy = DiffusionPolicy(config)
    policy.to(device)

    print("\nCreating EMA policy...")
    ema_policy = copy.deepcopy(policy)
    ema_decay = 0.999

    if cfg.use_compile and cfg.device.startswith("cuda"):
        print("\nCompiling policy with torch.compile()...")
        policy = torch.compile(policy)

    print("\nCreating preprocessor and postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset.meta.stats
    )

    optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Cache AMP constants outside the loop — avoids repeated torch.cuda API queries every step
    use_amp_cuda = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler(device='cuda') if use_amp_cuda else None
    pt_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp_cuda else None

    num_training_steps = len(train_dataloader) * cfg.num_epochs
    num_warmup_steps = int(num_training_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Created cosine scheduler with {num_warmup_steps} warmup steps and {num_training_steps} total steps.")

    if cfg.use_wandb:
        from dataclasses import asdict
        wandb.init(project=cfg.wandb_project, config=asdict(cfg), name=f"diffusion_lr{cfg.lr}")
        print("\nWeights & Biases logging enabled.")

    print(f"\nStarting training for {cfg.num_epochs} epochs on {device}...")

    best_val_loss = float("inf")
    best_success_rate = -1.0
    global_step = 0
    start_time = time.time()

    policy.train()
    for epoch in range(cfg.num_epochs):
        # All accumulators stay on GPU — zero CPU-GPU syncs during the epoch.
        total_train_loss = torch.zeros(1, device=device)
        total_grad_norm = torch.zeros(1, device=device)

        # Accumulators for the logging interval (averages over log_freq steps)
        interval_loss = torch.zeros(1, device=device)
        interval_grad_norm = torch.zeros(1, device=device)
        interval_count = 0

        epoch_start_time = time.time()
        interval_start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = preprocessor(batch)

            optimizer.zero_grad(set_to_none=True)

            if use_amp_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=pt_dtype):
                    loss, output_dict = policy.forward(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, output_dict = policy.forward(batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Update EMA policy (no-grad, GPU only)
            with torch.no_grad():
                for p, p_ema in zip(policy.parameters(), ema_policy.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            detached_loss = loss.detach()
            detached_grad_norm = grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm, device=device)

            total_train_loss += detached_loss
            total_grad_norm += detached_grad_norm
            interval_loss += detached_loss
            interval_grad_norm += detached_grad_norm
            interval_count += 1
            global_step += 1

            if batch_idx % cfg.log_freq == 0:
                # One sync per log_freq steps — loss and grad_norm .item() paid together.
                avg_step_loss = (interval_loss / interval_count).item()
                avg_step_grad_norm = (interval_grad_norm / interval_count).item()

                current_time = time.time()
                interval_duration = current_time - interval_start_time
                interval_bps = interval_count / interval_duration if interval_duration > 0 else 0.0

                print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Avg Loss: {avg_step_loss:.4f}, Avg Grad Norm: {avg_step_grad_norm:.4f}, {interval_bps:.1f} b/s")
                if cfg.use_wandb:
                    wandb.log({"train/step_loss": avg_step_loss, "train/step_grad_norm": avg_step_grad_norm, "train/lr": scheduler.get_last_lr()[0], "train/step_batches_per_sec": interval_bps}, step=global_step)

                # Reset interval accumulators
                interval_loss.zero_()
                interval_grad_norm.zero_()
                interval_count = 0
                interval_start_time = time.time()

        # Single sync per epoch: pull all epoch averages to CPU at once.
        n = len(train_dataloader)
        avg_train_loss = (total_train_loss / n).item()
        avg_grad_norm  = (total_grad_norm  / n).item()
        avg_lr = scheduler.get_last_lr()[0]
        batches_per_sec = n / (time.time() - epoch_start_time)
        print(f"==> Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Grad Norm: {avg_grad_norm:.4f} | LR: {avg_lr:.2e} | {batches_per_sec:.1f} batches/s")
        if cfg.use_wandb:
            wandb.log({"train/epoch_loss": avg_train_loss, "train/avg_grad_norm": avg_grad_norm, "train/lr": avg_lr, "train/batches_per_sec": batches_per_sec, "epoch": epoch}, step=global_step)

        # ==========================================
        # OFFLINE VALIDATION (Loss on val split)
        # ==========================================
        ema_policy.eval()
        total_val_loss = torch.zeros(1, device=device)
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=pt_dtype) if use_amp_cuda else nullcontext()
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                val_batch = preprocessor(val_batch)

                with amp_ctx:
                    v_loss, _ = ema_policy.forward(val_batch)

                total_val_loss += v_loss.detach()

        avg_val_loss = (total_val_loss / len(val_dataloader)).item()  # single sync
        print(f"==> Epoch {epoch+1} Average Val Loss: {avg_val_loss:.4f}")
        if cfg.use_wandb:
            wandb.log({"eval/val_loss": avg_val_loss, "epoch": epoch}, step=global_step)

        # Save best model based on validation loss
        os.makedirs(cfg.save_dir, exist_ok=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"🌟 New best model (val loss)! Saving to {os.path.join(cfg.save_dir, 'best_model')}")
            ema_policy.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
            preprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
            postprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model"))

        # Save latest model every epoch
        ema_policy.save_pretrained(os.path.join(cfg.save_dir, "latest_model"))
        preprocessor.save_pretrained(os.path.join(cfg.save_dir, "latest_model"))
        postprocessor.save_pretrained(os.path.join(cfg.save_dir, "latest_model"))

        # --- Evaluation Rollouts (sim environments only) ---
        if (epoch + 1) % cfg.eval_freq == 0 or (epoch + 1) == cfg.num_epochs:
            print(f"\n--- Running Evaluation Rollouts for {cfg.num_eval_episodes} episodes ---")
            env_id = "gym_pusht/PushT-v0" if "pusht" in cfg.dataset_id else cfg.dataset_id
            success_rate, ever_success_rate, avg_max_coverage, video_frames = rollout_and_evaluate(
                ema_policy, env_id, cfg.num_eval_episodes, device, preprocessor, postprocessor
            )
            print(f"==> Epoch {epoch+1} | Success: {success_rate*100:.1f}% | Ever: {ever_success_rate*100:.1f}% | Coverage: {avg_max_coverage:.4f}")

            if cfg.use_wandb:
                eval_metrics = {
                    "eval/success_rate": success_rate,
                    "eval/ever_success_rate": ever_success_rate,
                    "eval/avg_max_coverage": avg_max_coverage,
                    "epoch": epoch + 1,
                }
                if len(video_frames) > 0:
                    import numpy as np
                    video_array = np.stack(video_frames)                    # (T, H, W, C)
                    video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
                    eval_metrics["eval/video"] = wandb.Video(video_array, fps=cfg.fps, format="mp4")
                wandb.log(eval_metrics, step=global_step)

            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                print(f"🌟 New best model (rollout)! Saving to {os.path.join(cfg.save_dir, 'best_model_rollout')}")
                ema_policy.save_pretrained(os.path.join(cfg.save_dir, "best_model_rollout"))
                preprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model_rollout"))
                postprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model_rollout"))

        policy.train()

    print("Training complete!")

if __name__ == "__main__":
    main()
