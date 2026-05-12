import os
import torch
import gymnasium as gym
import gym_pusht
import imageio
import argparse
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained Diffusion policy.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/diffusion_pusht/best_model")
    parser.add_argument("--dataset_id", type=str, default="lerobot/pusht")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--output_video", type=str, default="inference_diffusion.mp4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading dataset {args.dataset_id} to extract stats...")
    dataset = LeRobotDataset(args.dataset_id)
    
    print(f"Loading policy from {args.checkpoint_dir}...")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint_dir)
    policy.to(device)
    policy.eval()

    print("Creating preprocessor...")
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset.meta.stats
    )

    env_id = "gym_pusht/PushT-v0" if "pusht" in args.dataset_id else args.dataset_id
    print(f"Starting {args.num_episodes} evaluation episodes in {env_id}...")
    
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.NameNotFound:
        print(f"\n⚠️ Environment {env_id} not found.")
        return

    successes = []
    best_video_frames = []

    with torch.no_grad():
        for ep in range(args.num_episodes):
            obs, info = env.reset()
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
                        T.Resize((96, 96)),
                        T.ToTensor(),
                    ])
                    img_tensor = transform(frame).to(device)
                    batch["observation.image"] = img_tensor.unsqueeze(0)
                
                if "observation.state" in policy.config.input_features:
                    state_tensor = torch.from_numpy(obs[:2]).float().to(device)
                    batch["observation.state"] = state_tensor.unsqueeze(0)
                
                batch = preprocessor(batch)
                
                action = policy.select_action(batch)
                
                # Unnormalize action (MIN_MAX)
                min_val = torch.from_numpy(dataset.meta.stats['action']['min']).to(device)
                max_val = torch.from_numpy(dataset.meta.stats['action']['max']).to(device)
                unnorm_action = (action + 1) / 2 * (max_val - min_val) + min_val
                
                action_np = unnorm_action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                step_count += 1
                
                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame)
            
            is_success = info.get("is_success", False) or info.get("success", False) or reward > 0.9
            successes.append(1.0 if is_success else 0.0)
            print(f"Episode {ep+1} finished after {step_count} steps. Success: {is_success}")
            
            if len(best_video_frames) == 0 or (is_success and sum(successes) == 1):
                best_video_frames = ep_frames

    env.close()
    
    print(f"\nFinal Success Rate: {sum(successes)/args.num_episodes * 100:.1f}%")
    
    if best_video_frames:
        print(f"Saving video of rollout to {args.output_video}...")
        imageio.mimsave(args.output_video, best_video_frames, fps=10)

if __name__ == "__main__":
    main()
