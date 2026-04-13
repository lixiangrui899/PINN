from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.zeros(1, device="cuda")
        probe = probe + 1
        _ = float(probe.item())
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_train_val_split(frame: pd.DataFrame, holdout_position: int, validation_position: int, group_column: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if group_column is None:
        test = frame[frame["position_id"] == holdout_position].copy()
        val = frame[frame["position_id"] == validation_position].copy()
        train = frame[(frame["position_id"] != holdout_position) & (frame["position_id"] != validation_position)].copy()
        return train, val, test

    group_lookup = frame.groupby(group_column)["position_id"].first().reset_index()
    test_groups = set(group_lookup[group_lookup["position_id"] == holdout_position][group_column].tolist())
    val_groups = set(group_lookup[group_lookup["position_id"] == validation_position][group_column].tolist())
    test = frame[frame[group_column].isin(test_groups)].copy()
    val = frame[frame[group_column].isin(val_groups)].copy()
    train = frame[~frame[group_column].isin(test_groups | val_groups)].copy()
    return train, val, test
