import torch
import gymnasium as gym
import gym_pusht
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = "lerobot/pusht"
    checkpoint_dir = "checkpoints/diffusion_pusht/best_model"

    print("Loading dataset stats and policy...")
    dataset = LeRobotDataset(dataset_id)
    policy = DiffusionPolicy.from_pretrained(checkpoint_dir)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset.meta.stats
    )

    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    obs, info = env.reset()

    # Get single frame
    frame = env.render()
    
    # Preprocess
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((96, 96)),
        T.ToTensor(),
    ])
    img_tensor = transform(frame).to(device).unsqueeze(0)
    state_tensor = torch.from_numpy(obs[:2]).float().to(device).unsqueeze(0)

    batch = {
        "observation.image": img_tensor,
        "observation.state": state_tensor
    }

    print("\n--- Raw Inputs ---")
    print(f"State (agent position): {obs[:2]}")
    
    batch = preprocessor(batch)
    print(f"Normalized State: {batch['observation.state'].cpu().numpy()}")

    with torch.no_grad():
        action = policy.select_action(batch)
        print("\n--- Model Output (from select_action) ---")
        print(f"Raw Output Action (Normalized): {action.cpu().numpy()}")
        
        # Look at the whole generated queue!
        chunk = policy._queues["action"]
        chunk_np = torch.stack(list(chunk), dim=0).cpu().numpy()
        print(f"\nRemaining Cached Actions in Queue (Normalized) - Shape {chunk_np.shape}:")
        print(chunk_np)

        print(f"\n--- Dataset Action Stats ---")
        print(f"Mean: {dataset.meta.stats['action']['mean']}")
        print(f"Std: {dataset.meta.stats['action']['std']}")
        print(f"Min: {dataset.meta.stats['action']['min']}")
        print(f"Max: {dataset.meta.stats['action']['max']}")

        # Manual unnormalization attempt
        mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(device)
        std = torch.from_numpy(dataset.meta.stats['action']['std']).to(device)
        unnorm_action = action * std + mean
        print(f"\n--- Manually Unnormalized Action ---")
        print(f"Unnormalized Action: {unnorm_action.cpu().numpy()}")

if __name__ == "__main__":
    main()
