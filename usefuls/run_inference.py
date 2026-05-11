import os
import torch
import gymnasium as gym
import gym_pusht
import imageio
import argparse
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained ACT policy.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/act_pusht/best_model",
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="lerobot/pusht",
        help="Dataset ID to get stats from for preprocessing.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="inference_rollout.mp4",
        help="Path to save the output video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--temporal_ensemble_coeff",
        type=float,
        default=None,
        help="Override temporal ensemble coefficient. If None, uses value from checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading dataset {args.dataset_id} to extract stats...")
    dataset = LeRobotDataset(args.dataset_id)
    
    print(f"Loading policy from {args.checkpoint_dir}...")
    kwargs = {}
    if args.temporal_ensemble_coeff is not None:
        kwargs["temporal_ensemble_coeff"] = args.temporal_ensemble_coeff
        kwargs["n_action_steps"] = 1  # Must be 1 when using temporal ensembling
    policy = ACTPolicy.from_pretrained(args.checkpoint_dir, **kwargs)
    policy.to(device)
    policy.eval()

    print("Creating preprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset.meta.stats
    )

    env_id = "gym_pusht/PushT-v0" if "pusht" in args.dataset_id else args.dataset_id
    print(f"Creating environment {env_id}...")
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym.error.NameNotFound:
        print(f"⚠️ Environment {env_id} not found. Exiting.")
        return

    successes = []
    best_video_frames = []

    print(f"Running {args.num_episodes} episodes...")
    with torch.no_grad():
        for ep in range(args.num_episodes):
            obs, info = env.reset()
            # Filter initial state to be within dataset distribution [120, 380]
            # assuming obs[:2] is the agent position.
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
                
                # Handle images
                if "observation.image" in policy.config.input_features:
                    import torchvision.transforms as T
                    transform = T.Compose([
                        T.ToPILImage(),
                        T.Resize((96, 96)),
                        T.ToTensor(),
                    ])
                    img_tensor = transform(frame).to(device).unsqueeze(0)
                    batch["observation.image"] = img_tensor
                
                # Handle state
                if "observation.state" in policy.config.input_features:
                     state_tensor = torch.from_numpy(obs[:2]).float().to(device).unsqueeze(0)
                     batch["observation.state"] = state_tensor
                # Preprocess
                batch = preprocessor(batch)
                
                # select_action
                action = policy.select_action(batch)
                
                # Unnormalize action
                mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(device)
                std = torch.from_numpy(dataset.meta.stats['action']['std']).to(device)
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
            print(f"Episode {ep+1} complete. Success: {is_success}")

            # Save the video of the first successful rollout (or the last rollout if none succeed)
            if len(best_video_frames) == 0 or (is_success and sum(successes) == 1):
                best_video_frames = ep_frames

    env.close()
    
    success_rate = sum(successes) / args.num_episodes
    print(f"Inference complete. Success Rate: {success_rate * 100:.1f}%")

    if len(best_video_frames) > 0:
        print(f"Saving video to {args.output_video}...")
        imageio.mimsave(args.output_video, best_video_frames, fps=10)
    else:
        print("No frames recorded to save video.")

if __name__ == "__main__":
    main()
