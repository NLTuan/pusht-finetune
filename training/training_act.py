import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler

from lerobot.policies.act import ACTPolicy
from lerobot.policies.act import ACTConfig

action_chunk_size = 50

dataset = LeRobotDataset("lerobot/pusht")

sampler = EpisodeAwareSampler(
    dataset.meta.episodes["dataset_from_index"],
    dataset.meta.episodes["dataset_to_index"],
    drop_n_last_frames=action_chunk_size,
    shuffle=True,
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
)


print(f"Total frames in dataset: {len(dataset)}")
print(f"Total episodes: {dataset.num_episodes}")

batch = next(iter(dataloader))
print("\nBatch inspection:")
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: type={type(value)}")