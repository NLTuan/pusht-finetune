import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

def main():
    parser = argparse.ArgumentParser(description="Visualize LeRobot Dataset Metadata")
    parser.add_argument("--repo_id", type=str, default="lerobot/pusht", help="Hugging Face repo ID of the dataset")
    parser.add_argument("--horizon", type=int, default=None, help="Action chunk size / horizon (e.g., 16 or 50)")
    parser.add_argument("--n_obs_steps", type=int, default=None, help="Observation history size (e.g., 2)")
    parser.add_argument("--sample_batch", action="store_true", help="Sample a single batch from a DataLoader and print its stats")
    args = parser.parse_args()

    print(f"Loading metadata for dataset: {args.repo_id}...")
    
    # Optional: We just instantiate a basic dataset first to get the FPS if we need to build timestamps
    # but LeRobotDataset fetches metadata dynamically.
    temp_dataset = LeRobotDataset(args.repo_id)
    fps = temp_dataset.meta.fps
    
    delta_timestamps = None
    if args.horizon is not None or args.n_obs_steps is not None:
        horizon = args.horizon if args.horizon else 1
        n_obs = args.n_obs_steps if args.n_obs_steps else 1
        
        delta_timestamps = {
            "action": [t / fps for t in range(horizon)]
        }
        
        # Add history for any observation keys that exist
        for key in temp_dataset.meta.features:
            if key.startswith("observation"):
                delta_timestamps[key] = [t / fps for t in range(-n_obs + 1, 1)]
                
        print(f"\nApplying delta_timestamps (Horizon={horizon}, Obs_Steps={n_obs})")
        
    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)

    fps = dataset.meta.fps
    total_frames = len(dataset)
    num_episodes = dataset.num_episodes
    
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Total Frames:   {total_frames}")
    print(f"Total Episodes: {num_episodes}")
    print(f"FPS:            {fps}")

    # Calculate timeframe features
    total_sec = total_frames / fps
    print(f"Total Duration: {total_sec:.2f} seconds ({total_sec/60:.2f} minutes)")
    
    if hasattr(dataset.meta, 'episodes') and "length" in dataset.meta.episodes:
        lengths = dataset.meta.episodes["length"].numpy()
        print(f"Avg Ep Length:  {lengths.mean():.1f} frames ({lengths.mean()/fps:.2f} sec)")
        print(f"Min Ep Length:  {lengths.min()} frames ({lengths.min()/fps:.2f} sec)")
        print(f"Max Ep Length:  {lengths.max()} frames ({lengths.max()/fps:.2f} sec)")

    print("\n" + "="*50)
    print("FEATURES (Shapes & Types)")
    if delta_timestamps:
        print(f"Note: Shapes are multiplied by the timeframe limits you set!")
    print("="*50)
    
    # Explicitly highlight action and state sizes
    if "action" in dataset.meta.features:
        print(f"⚡ Action Size (DoF): {dataset.meta.features['action'].get('shape', ['Unknown'])[0]}")
    if "observation.state" in dataset.meta.features:
        print(f"⚡ State Size: {dataset.meta.features['observation.state'].get('shape', ['Unknown'])[0]}")
    print("-" * 50)

    for key, ft in dataset.meta.features.items():
        print(f"🔑 {key}:")
        for k, v in ft.items():
            print(f"   - {k.capitalize()}: {v}")

    if args.sample_batch:
        from torch.utils.data import DataLoader
        from lerobot.datasets.sampler import EpisodeAwareSampler

        print("\n" + "="*50)
        print("SAMPLE BATCH STATISTICS (Batch Size: 4)")
        print("="*50)
        
        drop_n = args.horizon if args.horizon else 0
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=drop_n,
            shuffle=True,
        )
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
        
        batch = next(iter(dataloader))
        
        for key, tensor in batch.items():
            if hasattr(tensor, 'shape'):
                print(f"📦 {key}:")
                print(f"   - Shape: {list(tensor.shape)}")
                print(f"   - Dtype: {tensor.dtype}")
                if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
                    print(f"   - Min:   {tensor.min().item():.3f}")
                    print(f"   - Max:   {tensor.max().item():.3f}")
                    print(f"   - Mean:  {tensor.float().mean().item():.3f}")
            elif isinstance(tensor, (list, tuple)):
                print(f"📦 {key}:")
                print(f"   - Type: {type(tensor).__name__} (len={len(tensor)})")
                if len(tensor) > 0:
                    # Truncate string if it's too long
                    val_str = str(tensor[0])
                    if len(val_str) > 50:
                        val_str = val_str[:47] + "..."
                    print(f"   - First Element: {val_str}")
            else:
                print(f"📦 {key}:")
                print(f"   - Type: {type(tensor)}")

if __name__ == "__main__":
    main()
