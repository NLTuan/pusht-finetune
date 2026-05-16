import os
import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
import time

def evaluate_model(model_path, val_dataloader, device):
    if not os.path.exists(model_path):
        print(f"Path not found: {model_path}")
        return None
        
    print(f"\nLoading model from {model_path}...")
    policy = ACTPolicy.from_pretrained(model_path)
    policy.to(device)
    policy.eval()
    
    # Create processors (using dataset stats from the current dataset)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=val_dataloader.dataset.meta.stats
    )
    
    total_val_loss = torch.zeros(1, device=device)
    use_amp = True
    pt_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=pt_dtype) if use_amp else nullcontext()
    
    print("Evaluating...")
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
            
    avg_val_loss = (total_val_loss / len(val_dataloader)).item()
    return avg_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_id = "NLTuan/red_blue_cleaned_extra"
    action_chunk_size = 50
    fps = 10
    batch_size = 8
    
    delta_timestamps = {"action": [t / fps for t in range(action_chunk_size)]}
    
    print("Loading dataset (NO augmentations)...")
    val_dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, image_transforms=None)
    
    total_episodes = val_dataset.num_episodes
    val_split_idx = int(total_episodes * 0.9)
    
    val_sampler = EpisodeAwareSampler(
        val_dataset.meta.episodes["dataset_from_index"][val_split_idx:],
        val_dataset.meta.episodes["dataset_to_index"][val_split_idx:],
        drop_n_last_frames=action_chunk_size,
        shuffle=False,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True,
    )
    
    base_dir = "checkpoints/act_red_blue"
    best_path = os.path.join(base_dir, "best_model")
    latest_path = os.path.join(base_dir, "latest_model")
    
    best_loss = evaluate_model(best_path, val_dataloader, device)
    latest_loss = evaluate_model(latest_path, val_dataloader, device)
    
    print("\n" + "="*40)
    print("FINAL RESULTS (No Augmentation)")
    print("="*40)
    if best_loss is not None:
        print(f"Best Model Loss:   {best_loss:.4f}")
    if latest_loss is not None:
        print(f"Latest Model Loss: {latest_loss:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
