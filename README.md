# PushT Finetune (LeRobot)

This repository contains optimized, production-ready training loops and utility scripts for training modern robotics policies (ACT, Diffusion, PI0) using the [LeRobot](https://github.com/huggingface/lerobot) framework.

## 🚀 Getting Started

This project uses `uv` for lightning-fast package management.

### 1. Setup Environment
Ensure your environment is set up and dependencies are installed.

First, clone your necessary third-party repositories (like LeRobot):
```bash
bash setup_third_party.sh
```

Then, install all project dependencies:
```bash
# If using uv for the first time in this repo
uv sync
```

### 2. Login to Weights & Biases
All of our training scripts are fully integrated with Weights & Biases (`wandb`) to log training loss, offline validation metrics, and online simulation rollouts (videos!). Before running a training script, make sure to authenticate:
```bash
uv run wandb login
```

## 🛠️ Commands & Usage

### Inspecting Datasets

Before training, it is highly recommended to visualize the dataset shapes, feature dimensions, and timeframes. We have a dedicated utility script for this:

```bash
# Basic overview of a dataset (default is lerobot/pusht)
uv run usefuls/visualize_dataset_metadata.py --repo_id lerobot/pusht

# See how time limits (chunking/history) affect your tensor shapes
uv run usefuls/visualize_dataset_metadata.py --horizon 16 --n_obs_steps 2

# Sample an actual dataloader batch and print out exact tensor shapes, dtypes, mins, and maxes
uv run usefuls/visualize_dataset_metadata.py --sample_batch
```

### Training Policies

Each policy has its own dedicated, optimized training loop. They all feature dynamic feature extraction, `torch.compile`, automatic mixed precision (AMP), and integrated simulation rollouts.

#### ACT Policy
```bash
uv run training/training_act.py
```

#### Diffusion Policy
```bash
uv run training/training_diffusion.py
```

#### PI0 Policy (Flow Matching)
```bash
uv run training/training_pi.py
```

*(Note: To change hyperparameters like `batch_size`, `num_epochs`, `eval_freq`, or the `dataset_id`, simply modify the `TrainConfig` dataclass located at the very top of each training script!)*
