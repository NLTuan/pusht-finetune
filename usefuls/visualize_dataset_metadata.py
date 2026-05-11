import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

def main():
    parser = argparse.ArgumentParser(description="Visualize LeRobot Dataset Metadata")
    parser.add_argument("--repo_id", type=str, default="lerobot/pusht", help="Hugging Face repo ID of the dataset")
    args = parser.parse_args()

    print(f"Loading metadata for dataset: {args.repo_id}...")
    dataset = LeRobotDataset(args.repo_id)

    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Total Frames:   {len(dataset)}")
    print(f"Total Episodes: {dataset.num_episodes}")
    print(f"FPS:            {dataset.meta.fps}")

    print("\n" + "="*50)
    print("FEATURES (Shapes & Types)")
    print("="*50)
    for key, ft in dataset.meta.features.items():
        print(f"🔑 {key}:")
        for k, v in ft.items():
            print(f"   - {k.capitalize()}: {v}")



if __name__ == "__main__":
    main()
