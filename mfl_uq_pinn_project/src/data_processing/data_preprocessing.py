"""
Normalization and dataset splitting utilities.

数据归一化与数据集划分工具：支持 Z-score、Min-Max 和固定随机种子分割。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ZScoreStats:
    """Store mean and std for z-score normalization / 存储均值与标准差"""

    mean: np.ndarray
    std: np.ndarray


@dataclass
class MinMaxStats:
    """Store min and max for min-max scaling / 存储最小值和最大值"""

    min_vals: np.ndarray
    max_vals: np.ndarray


def zscore_normalize(signals: np.ndarray) -> Tuple[np.ndarray, ZScoreStats]:
    """
    Apply per-sample z-score normalization to 2D array (num_samples, signal_length).

    对信号矩阵逐样本执行 Z-score 归一化，消除量纲差异。
    """

    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True) + 1e-12
    normalized = (signals - mean) / std
    return normalized, ZScoreStats(mean=mean, std=std)


def minmax_normalize_params(
    params: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray
) -> Tuple[np.ndarray, MinMaxStats]:
    """
    Normalize defect parameters to [0, 1] using provided min/max.

    使用指定的最小/最大值对缺陷参数执行 Min-Max 归一化。
    """

    span = (max_vals - min_vals) + 1e-12
    normalized = (params - min_vals) / span
    return normalized, MinMaxStats(min_vals=min_vals, max_vals=max_vals)


def split_dataset(
    signals: np.ndarray,
    params: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Split dataset into train/val/test using fixed seed.

    按比例切分数据集（默认 7:2:1），使用固定种子保证可复现。
    """

    assert signals.shape[0] == params.shape[0], "Signal and param counts must match."
    num_samples = signals.shape[0]
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)

    idx_train = indices[:n_train]
    idx_val = indices[n_train : n_train + n_val]
    idx_test = indices[n_train + n_val :]

    def select(idx):
        return signals[idx], params[idx]

    train = select(idx_train)
    val = select(idx_val)
    test = select(idx_test)
    return train, val, test
