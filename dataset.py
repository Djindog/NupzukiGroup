import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npy_dict(file_path: str):
    loaded = np.load(file_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {k: loaded[k] for k in loaded.files}
    raise ValueError(f"Unsupported data format in {file_path}")


class InstancePointCloudDataset(Dataset):
    """
    Dataset for instance segmentation.

    Returns:
      - features: [9, N]
      - instance_labels: [N], background=0
      - obj_labels: [N], binary foreground/background
      - scene_path
    """

    def __init__(self, data_dir: str, split: str = "train", seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15):
        self.data_dir = data_dir
        self.split = split

        all_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
        if len(all_files) == 0:
            raise ValueError(f"No *.npy files found under: {data_dir}")

        rng = np.random.default_rng(seed)
        rng.shuffle(all_files)

        n = len(all_files)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.files = all_files[:train_end]
        elif split == "val":
            self.files = all_files[train_end:val_end]
        elif split == "test":
            self.files = all_files[val_end:]
        elif split == "all":
            self.files = all_files
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = _load_npy_dict(file_path)

        xyz = np.asarray(data["xyz"], dtype=np.float32)
        rgb = np.asarray(data["rgb"], dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        normal = np.asarray(data["normal"], dtype=np.float32)

        if "instance_labels" in data:
            instance_labels = np.asarray(data["instance_labels"], dtype=np.int64)
        else:
            instance_labels = np.asarray(data["is_mesh"], dtype=np.int64)

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        radius = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if radius > 1e-8:
            xyz = xyz / radius

        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = np.divide(normal, normal_norm, out=normal, where=normal_norm != 0)

        features = np.concatenate([xyz, rgb, normal], axis=1).T

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "instance_labels": torch.tensor(instance_labels, dtype=torch.long),
            "scene_path": file_path,
        }


# Backward-compatible alias
TestPointCloudDataset = InstancePointCloudDataset
PointCloudBinarySegDataset = InstancePointCloudDataset
