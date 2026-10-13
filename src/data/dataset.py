"""
Scalable Sign Language Dataset
================================
PyTorch Dataset and DataLoader utilities for training on dynamically
discovered gloss classes. Supports lazy-loading, temporal padding/truncation,
augmentation, and automatic train/val/test splitting.

Compatible with both the original 3-class MP_Data structure and the
WLASL pipeline output (hundreds or thousands of classes).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple, List

from src.features.normalizer import LandmarkNormalizer
from src.features.velocity import VelocityComputer


class SignLanguageDataset(Dataset):
    """Lazy-loading PyTorch Dataset for sign language landmark sequences.

    Features:
        - Automatic folder discovery — no hardcoded action list
        - Lazy .npy loading — only loads data when __getitem__ is called
        - Temporal padding/truncation to uniform sequence length
        - Optional data augmentation via SequenceAugmentor
        - Label map persistence (label_map.json)
    """

    def __init__(
        self,
        data_dir: str,
        max_frames: int = 60,
        augmentor=None,
    ):
        """
        Parameters
        ----------
        data_dir : str
            Root directory containing one sub-folder per gloss class,
            each containing .npy files.
        max_frames : int
            Fixed temporal length. Sequences are padded or truncated to this.
        augmentor : SequenceAugmentor or None
            Optional augmentation applied during __getitem__.
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.augmentor = augmentor
        self.file_paths: List[str] = []
        self.labels: List[str] = []

        # ---- Automatic folder discovery ----
        classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        if not classes:
            raise ValueError(f"No class directories found in {data_dir}")

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(classes)

        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    self.file_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(cls)

        self.encoded_labels = self.label_encoder.transform(self.labels)
        self.num_classes = len(classes)
        self.normalizer = LandmarkNormalizer()
        self.classes = classes

        print(f"[Dataset] Loaded {len(self.file_paths)} samples across {self.num_classes} classes from {data_dir}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (sequence_tensor, label_tensor, actual_length).

        The actual_length is needed by pad_collate_fn to create attention masks.
        """
        sequence = np.load(self.file_paths[idx])

        # If raw 258-dim MediaPipe (pose 132 + lh 63 + rh 63), normalize and compute 454-dim features
        if sequence.shape[1] == 258:
            T = sequence.shape[0]
            p_full = sequence[:, :132].reshape(T, 33, 4)[:, :25, :].reshape(T, 100)
            lh = sequence[:, 132:195]
            rh = sequence[:, 195:258]

            norm_frames = [self.normalizer.normalize_frame(lh[i], rh[i], p_full[i]) for i in range(T)]
            norm_arr = np.array(norm_frames, dtype=np.float32)

            lh_p = (np.abs(lh).sum(axis=1, keepdims=True) > 1e-4).astype(np.float32)
            rh_p = (np.abs(rh).sum(axis=1, keepdims=True) > 1e-4).astype(np.float32)
            flags = np.concatenate([lh_p, rh_p], axis=1)

            sequence = VelocityComputer.build_batch_features(norm_arr, flags)

        # Optional augmentation (spatial noise, speed variation, etc.)
        if self.augmentor is not None:
            sequence = self.augmentor(sequence)

        seq_len = sequence.shape[0]
        feat_dim = sequence.shape[1]
        actual_len = min(seq_len, self.max_frames)

        # Temporal padding / truncation
        if seq_len > self.max_frames:
            sequence = sequence[:self.max_frames]
        elif seq_len < self.max_frames:
            padding = np.zeros((self.max_frames - seq_len, feat_dim), dtype=np.float32)
            sequence = np.vstack((sequence, padding))

        sequence_tensor = torch.FloatTensor(sequence)
        label_tensor = torch.LongTensor([self.encoded_labels[idx]])[0]

        return sequence_tensor, label_tensor, actual_len

    def save_label_map(self, path: str):
        """Save the class-to-index mapping for inference."""
        mapping = {cls: int(idx) for cls, idx in
                   zip(self.classes, self.label_encoder.transform(self.classes))}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)


# --------------------------------------------------------------------------
# Collation function for variable-length sequences
# --------------------------------------------------------------------------

def pad_collate_fn(batch):
    """Custom collate that also returns a boolean attention mask.

    Returns
    -------
    sequences : Tensor (B, T, D)
    labels    : Tensor (B,)
    mask      : Tensor (B, T) — True for real frames, False for padding
    """
    sequences, labels, lengths = zip(*batch)

    sequences = torch.stack(sequences, dim=0)
    labels = torch.stack(labels, dim=0)

    max_len = sequences.shape[1]
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return sequences, labels, mask


# --------------------------------------------------------------------------
# DataLoader factory
# --------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str,
    max_frames: int = 60,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    augmentor=None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SignLanguageDataset]:
    """Create train/val/test DataLoaders from a flat directory of .npy files.

    Returns
    -------
    train_loader, val_loader, test_loader, full_dataset
    """
    dataset = SignLanguageDataset(
        data_dir=data_dir,
        max_frames=max_frames,
        augmentor=augmentor,
    )

    total = len(dataset)
    train_n = int(total * train_split)
    val_n = int(total * val_split)
    test_n = total - train_n - val_n

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate_fn,
    )

    return train_loader, val_loader, test_loader, dataset
