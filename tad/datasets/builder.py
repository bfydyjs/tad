from collections.abc import Sequence

import torch
from torch.utils.data.dataloader import default_collate

from tad.utils.registry import Registry

DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")


class Pipeline:
    """Compose multiple transforms sequentially to form a data pipeline."""

    def __init__(self, transforms):
        self.transforms = []
        if transforms is None:
            transforms = []

        for transform in transforms:
            if isinstance(transform, dict):
                self.transforms.append(TRANSFORMS.build(transform))
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be callable or dict, got {type(transform)}")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict."""
    return DATASETS.build(cfg, default_args)


def build_dataloader(
    dataset, batch_size, rank, world_size, shuffle=False, drop_last=False, **kwargs
):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    assert batch_size % world_size == 0, (
        f"batch size {batch_size} should be divided by world size {world_size}"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size // world_size,
        collate_fn=collate,
        pin_memory=True,
        sampler=sampler,
        **kwargs,
    )
    return dataloader


def collate(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    gpu_stack_keys = ["inputs", "masks"]

    collate_data = {}
    for key in batch[0]:
        if key in gpu_stack_keys:
            collate_data[key] = default_collate([sample[key] for sample in batch])
        else:
            collate_data[key] = [sample[key] for sample in batch]
    return collate_data
