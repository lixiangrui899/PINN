from __future__ import annotations

import math

import numpy as np
import torch


def gaussian_nll(targets: np.ndarray, mean: np.ndarray, logvar: np.ndarray) -> float:
    variance = np.exp(np.clip(logvar, -6.0, 6.0))
    return float(0.5 * np.mean(((targets - mean) ** 2) / variance + np.log(variance)))


def prediction_interval(mean: np.ndarray, logvar: np.ndarray, z_score: float) -> tuple[np.ndarray, np.ndarray]:
    std = np.sqrt(np.exp(np.clip(logvar, -6.0, 6.0)))
    return mean - z_score * std, mean + z_score * std


def picp(targets: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    covered = ((targets >= lower) & (targets <= upper)).astype(np.float32)
    return float(covered.mean())


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def calibration_curve(targets: np.ndarray, mean: np.ndarray, logvar: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    std = np.sqrt(np.exp(np.clip(logvar, -6.0, 6.0)))
    z_scores = np.abs(targets - mean) / np.maximum(std, 1e-6)
    nominal_levels = np.linspace(0.1, 0.99, bins)
    normal = torch.distributions.Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
    curve = []
    for level in nominal_levels:
        quantile = (1.0 + float(level)) / 2.0
        threshold = float(normal.icdf(torch.tensor(quantile)))
        empirical = float((z_scores <= threshold).mean())
        curve.append({"nominal": float(level), "empirical": empirical})
    return curve
