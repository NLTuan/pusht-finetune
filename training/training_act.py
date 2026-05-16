import os
import torch
import torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader
import wandb
import gymnasium as gym
from dataclasses import dataclass
import time
from huggingface_hub import HfApi, model_info
from huggingface_hub.utils import RevisionNotFoundError

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.transforms import ImageTransforms, ImageTransformsConfig
from transformers import get_cosine_schedule_with_warmup

@dataclass
class TrainConfig:
    dataset_id: str = "NLTuan/red_blue_cleaned_extra"
    action_chunk_size: int = 50
    fps: int = 10
    batch_size: int = 8
    lr: float = 5e-5
    weight_decay: float = 1e-4
    num_epochs: int = 20
    log_freq: int = 50
    val_freq: int = 0            # run val every N steps mid-epoch; 0 = epoch-end only
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    use_compile: bool = True
    num_workers: int = 4

    eval_freq: int = 20          # don't eval every epoch, it's slow
    num_eval_episodes: int = 20  # more reliable signal
    run_eval: bool = False       # set to False for real-life data with no simulator
    save_dir: str = "checkpoints/act_red_blue"
    use_wandb: bool = True
    wandb_project: str = "red-blue-act"
    hub_repo_id: str = "NLTuan/act-red-blue-policy"

def rollout_and_evaluate(policy, env_id, num_episodes, device, preprocessor, postprocessor):
    """Run simulation rollouts to evaluate the policy."""
    import imageio
    import numpy as np
    
    # NOTE: You may need to import your specific env registration here (e.g., import gym_pusht)
    import gym_pusht
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.NameNotFound:
        print(f"\n⚠️ Environment {env_id} not found. Skipping evaluation.")
        print("Please ensure your environment is registered with gymnasium.")
        return 0.0, []

    policy.eval()
    successes = []
    ever_successes = []
    max_coverages = []
    best_video_frames = []

    # Hoist these out of the inner loop — they are constant for the entire rollout.
    input_features = policy.config.input_features
    has_image = "observation.image" in input_features
    has_state = "observation.state" in input_features

    if has_image:
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((96, 96)),
            T.ToTensor(),
        ])

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, info = env.reset()
            # Filter initial state to be within dataset distribution [120, 380]
            # assuming obs[:2] is the agent position.
            reset_count = 0
            while (obs[0] < 120 or obs[0] > 380 or obs[1] < 120 or obs[1] > 380) and reset_count < 100:
                obs, info = env.reset()
                reset_count += 1
                
            done = False
            ever_succeeded = False
            current_max_coverage = 0.0
            
            # Clear observation history / queues in the policy
            if hasattr(policy, "reset"):
                policy.reset()
                
            ep_frames = []
            
            frame = env.render()
            if frame is not None:
                ep_frames.append(frame)
                
            # We assume a max step limit if the env doesn't truncate
            step_count = 0
            while not done and step_count < 1000:
                batch = {}
                
                if has_image:
                    img_tensor = transform(frame).to(device).unsqueeze(0)
                    batch["observation.image"] = img_tensor
                
                if has_state:
                    # Assume first 2 dims are agent pos
                    state_tensor = torch.from_numpy(obs[:2]).float().to(device).unsqueeze(0)
                    batch["observation.state"] = state_tensor
                
                # Preprocess (normalize)
                batch = preprocessor(batch)
                
                # select_action automatically handles action chunking history internally!
                action = policy.select_action(batch)
                
                # Unnormalize action via LeRobot postprocessor (UnnormalizerProcessorStep + CPU move)
                # This is the same pipeline used by SyncInferenceEngine and PolicyServer,
                # so the model is compatible with lerobot-rollout and async inference out of the box.
                action_np = postprocessor(action).squeeze(0).numpy()
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                current_max_coverage = max(current_max_coverage, reward)
                if reward > 0.9 or info.get("is_success", False) or info.get("success", False):
                    ever_succeeded = True
                done = terminated or truncated
                step_count += 1
                
                # Render frame for video and next step
                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame)
            
            is_success = info.get("is_success", False) or info.get("success", False) or reward > 0.9
            successes.append(1.0 if is_success else 0.0)
            ever_successes.append(1.0 if ever_succeeded else 0.0)
            max_coverages.append(current_max_coverage)
            
            # Save the video of the first successful rollout (or the last rollout if none succeed)
            if len(best_video_frames) == 0 or (is_success and sum(successes) == 1):
                best_video_frames = ep_frames
                
    env.close()
    policy.train()
    terminal_success_rate = sum(successes) / num_episodes
    ever_success_rate = sum(ever_successes) / num_episodes
    avg_max_coverage = sum(max_coverages) / num_episodes
    return terminal_success_rate, ever_success_rate, avg_max_coverage, best_video_frames

def run_validation(policy, val_dataloader, preprocessor, device, use_amp_cuda, pt_dtype):
    """Run the full validation loop and return average loss. Leaves policy in eval mode."""
    policy.eval()
    total_val_loss = torch.zeros(1, device=device)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=pt_dtype) if use_amp_cuda else nullcontext()
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
            val_batch = preprocessor(val_batch)
            with amp_ctx:
                actions_hat = policy.predict_action_chunk(val_batch)
                abs_err = torch.nn.functional.l1_loss(val_batch["action"], actions_hat, reduction="none")
                valid_mask = ~val_batch["action_is_pad"].unsqueeze(-1)
                v_loss = (abs_err * valid_mask).sum() / (valid_mask.sum() * abs_err.shape[-1]).clamp_min(1)
            total_val_loss += v_loss.detach()
    return (total_val_loss / len(val_dataloader)).item()

def main():
    cfg = TrainConfig()

    delta_timestamps = {"action": [t / cfg.fps for t in range(cfg.action_chunk_size)]}

    image_transforms = ImageTransforms(ImageTransformsConfig(enable=True))
    dataset = LeRobotDataset(cfg.dataset_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms)
    
    # Create a separate dataset instance for validation with NO augmentations.
    # This ensures a stable, clean validation loss curve.
    val_dataset = LeRobotDataset(cfg.dataset_id, delta_timestamps=delta_timestamps, image_transforms=None)

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
        persistent_workers=cfg.num_workers > 0,
    )

    # Validation Sampler (Episodes val_split_idx to end)
    val_sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"][val_split_idx:],
        dataset.meta.episodes["dataset_to_index"][val_split_idx:],
        drop_n_last_frames=cfg.action_chunk_size,
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

    print("\nInstantiating ACT policy with dynamic configs...")
    
    # Dynamically extract features from dataset
    input_features = {}
    for key, ft in dataset.meta.features.items():
        if key.startswith("observation.image"):
            # Dataset meta stores image shape as (H, W, C), but PolicyFeature expects (C, H, W)
            shape = (ft["shape"][2], ft["shape"][0], ft["shape"][1])
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
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
        n_action_steps=cfg.action_chunk_size,  # Must be 1 when using temporal ensembling
        # temporal_ensemble_coeff=0.01,
        kl_weight=10,
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

    use_amp_cuda = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler(device='cuda') if use_amp_cuda else None
    # Resolve AMP dtype once — avoids repeated torch.cuda.is_bf16_supported() calls in the loop
    pt_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp_cuda else None

    num_training_steps = len(train_dataloader) * cfg.num_epochs
    num_warmup_steps = int(num_training_steps * 0.02)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Created cosine scheduler with {num_warmup_steps} warmup steps and {num_training_steps} total steps.")

    if cfg.use_wandb:
        # We can pass the dataclass fields directly into wandb config
        from dataclasses import asdict
        wandb.init(project=cfg.wandb_project, config=asdict(cfg), name=f"act_red_blue_lr{cfg.lr}")
        print("\nWeights & Biases logging enabled.")

    print(f"\nStarting training for {cfg.num_epochs} epochs on {device}...")

    best_success_rate = -1.0
    best_val_loss = float('inf')
    global_step = 0
    # start_time kept for reference; per-epoch timing used for throughput metric
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
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Preprocess batch (normalize)
            # Note: image augmentation is handled by LeRobot's ImageTransforms in the DataLoader workers
            batch = preprocessor(batch)
            
            optimizer.zero_grad(set_to_none=True) # Faster than zero_grad()
            
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
            
            detached_loss = loss.detach()
            detached_grad_norm = grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm, device=device)
            
            total_train_loss += detached_loss
            total_grad_norm += detached_grad_norm
            
            interval_loss += detached_loss
            interval_grad_norm += detached_grad_norm
            interval_count += 1
            global_step += 1
            
            if batch_idx % cfg.log_freq == 0:
                # One sync per log_freq steps — pulls the average of the last interval.
                avg_step_loss = (interval_loss / interval_count).item()
                avg_step_grad_norm = (interval_grad_norm / interval_count).item()
                
                # Calculate interval batches per second
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

            # Mid-epoch validation — fires every val_freq steps when enabled.
            if cfg.val_freq > 0 and global_step % cfg.val_freq == 0:
                avg_val_loss = run_validation(policy, val_dataloader, preprocessor, device, use_amp_cuda, pt_dtype)
                print(f"  [Step {global_step}] Val Loss: {avg_val_loss:.4f}")
                if cfg.use_wandb:
                    wandb.log({"eval/val_loss": avg_val_loss}, step=global_step)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"  🌟 New best model! Saving to {os.path.join(cfg.save_dir, 'best_model')}")
                    policy.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
                    preprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
                    postprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
                policy.train()

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
        avg_val_loss = run_validation(policy, val_dataloader, preprocessor, device, use_amp_cuda, pt_dtype)
        print(f"==> Epoch {epoch+1} Average Val Loss: {avg_val_loss:.4f}")
        if cfg.use_wandb:
            wandb.log({"eval/val_loss": avg_val_loss, "epoch": epoch}, step=global_step)
            
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"🌟 New best model based on Val Loss! Saving to {os.path.join(cfg.save_dir, 'best_model')}")
            policy.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
            preprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
            postprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
            
        policy.train() # Set back to train mode

        # ==========================================
        # EVALUATION ROLLOUTS & CHECKPOINTING
        # ==========================================
        # Save latest model every epoch
        os.makedirs(cfg.save_dir, exist_ok=True)
        policy.save_pretrained(os.path.join(cfg.save_dir, "latest_model"))
        preprocessor.save_pretrained(os.path.join(cfg.save_dir, "latest_model"))
        postprocessor.save_pretrained(os.path.join(cfg.save_dir, "latest_model"))
        
        # Run online evaluation
        if cfg.run_eval and (epoch + 1) % cfg.eval_freq == 0:
            print(f"\n--- Running Evaluation Rollouts for {cfg.num_eval_episodes} episodes ---")
            
            # We map dataset ID to gym env ID if needed, e.g. lerobot/pusht -> gym_pusht/PushT-v0
            env_id = "gym_pusht/PushT-v0" if "pusht" in cfg.dataset_id else cfg.dataset_id
            
            success_rate, ever_success_rate, avg_max_coverage, video_frames = rollout_and_evaluate(policy, env_id, cfg.num_eval_episodes, device, preprocessor, postprocessor)
            print(f"==> Epoch {epoch+1} Terminal Success Rate: {success_rate * 100:.1f}%, Ever Success Rate: {ever_success_rate * 100:.1f}%, Avg Max Coverage: {avg_max_coverage:.4f}")
            
            if cfg.use_wandb:
                eval_metrics = {
                    "eval/success_rate": success_rate, 
                    "eval/ever_success_rate": ever_success_rate, 
                    "eval/avg_max_coverage": avg_max_coverage,
                    "epoch": epoch
                }
                
                if len(video_frames) > 0:
                    import numpy as np
                    # Wandb Video expects shape (time, channel, height, width)
                    vid_tensor = np.array(video_frames).transpose(0, 3, 1, 2)
                    eval_metrics["eval/rollout_video"] = wandb.Video(vid_tensor, fps=cfg.fps, format="mp4")
                    
                wandb.log(eval_metrics, step=global_step)
            
            # Save best model
            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                print(f"🌟 New best model! Saving to {os.path.join(cfg.save_dir, 'best_model')}")
                policy.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
                preprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
                postprocessor.save_pretrained(os.path.join(cfg.save_dir, "best_model"))
                
    if cfg.use_wandb:
        wandb.finish()
    print("\nTraining complete!")
    
    api = HfApi()
    repo_id = f"{cfg.hub_repo_id}-lr{cfg.lr}"
    
    # Create repo if it doesn't exist yet
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # Push best model to main branch
    print(f"Pushing best model to {repo_id} (branch: main)...")
    api.upload_folder(
        folder_path=os.path.join(cfg.save_dir, "best_model"),
        repo_id=repo_id,
        revision="main",
    )
    
    # Create 'latest' branch if it doesn't exist
    print(f"Checking if 'latest' branch exists for {repo_id}...")
    try:
        model_info(repo_id, revision="latest")
        print("'latest' branch already exists.")
    except RevisionNotFoundError:
        print("Creating 'latest' branch...")
        api.create_branch(repo_id, branch="latest")
        print("Created 'latest' branch.")
        
    # Push latest model to latest branch
    print(f"Pushing latest model to {repo_id} (branch: latest)...")
    api.upload_folder(
        folder_path=os.path.join(cfg.save_dir, "latest_model"),
        repo_id=repo_id,
        revision="latest",
    )

if __name__ == "__main__":
    main()