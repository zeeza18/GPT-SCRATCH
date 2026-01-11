"""
Dataset classes and data loading utilities for LLM training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


class TextDataset(Dataset):
    """
    Dataset for text generation tasks.

    Creates input-target pairs for next-token prediction.
    """

    def __init__(self, text_ids: List[int], context_length: int, stride: int = 1):
        """
        Initialize text dataset.

        Args:
            text_ids: List of token IDs from tokenized text
            context_length: Length of input sequences
            stride: Stride for sliding window (default: 1 for no overlap)
        """
        self.text_ids = text_ids
        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return (len(self.text_ids) - self.context_length) // self.stride

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training example.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with 'input_ids' and 'target_ids'
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length

        input_ids = torch.tensor(self.text_ids[start_idx:end_idx], dtype=torch.long)
        target_ids = torch.tensor(self.text_ids[start_idx + 1:end_idx + 1], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }


def create_dataloader(text_ids: List[int], batch_size: int = 4,
                     context_length: int = 256, stride: int = 128,
                     shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for text data.

    Args:
        text_ids: List of token IDs
        batch_size: Batch size for training
        context_length: Length of input sequences
        stride: Stride for sliding window
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    dataset = TextDataset(text_ids, context_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
    return dataloader
