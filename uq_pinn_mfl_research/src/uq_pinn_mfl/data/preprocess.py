from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from uq_pinn_mfl.config import ProjectContext


def _stable_token(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _normalize_signal(signal: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return signal.astype(np.float32)
    if mode == "zscore_per_channel":
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return ((signal - mean) / std).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def load_signal_csv(path: str | Path, channel_columns: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, usecols=channel_columns, encoding="utf-8-sig")
    return frame.to_numpy(dtype=np.float32).T


def preprocess_dataset(context: ProjectContext, rebuild: bool = False, limit_files: int | None = None) -> dict[str, Path]:
    if not context.dataset_index_path.exists():
        raise FileNotFoundError("dataset_index.csv is missing. Run Stage A audit first.")

    dataset_index = pd.read_csv(context.dataset_index_path)
    if limit_files is not None:
        dataset_index = dataset_index.head(limit_files).copy()

    cfg = context.config["preprocessing"]
    channel_columns = cfg["channel_columns"]
    normalization = cfg["normalization"]
    cache_dir = context.cache_dir / "signals"
    pseudo_dir = context.cache_dir / "pseudo_targets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for _, row in dataset_index.iterrows():
        cache_token = _stable_token(row["relative_path"])
        cache_path = cache_dir / f"{cache_token}.npz"
        signal: np.ndarray | None = None
        if rebuild or not cache_path.exists():
            signal = load_signal_csv(row["path"], channel_columns)
            signal = _normalize_signal(signal, normalization)
            np.savez_compressed(cache_path, signal=signal)
        elif limit_files is not None:
            loaded = np.load(cache_path)
            signal = loaded["signal"]

        if signal is None:
            loaded = np.load(cache_path)
            signal = loaded["signal"]

        records.append(
            {
                **row.to_dict(),
                "cache_path": str(cache_path),
                "num_channels": int(signal.shape[0]),
                "num_samples": int(signal.shape[1]),
                "pseudo_target_group": f"{row['defect_unit_id']}__{row['orientation']}__{row['acquisition_tag']}",
            }
        )

    preprocessed = pd.DataFrame(records)
    pseudo_map = build_pseudo_targets(preprocessed, pseudo_dir)
    preprocessed["pseudo_target_path"] = preprocessed["pseudo_target_group"].map(pseudo_map)
    preprocessed_path = context.reports_dir / "preprocessed_index.csv"
    preprocessed.to_csv(preprocessed_path, index=False, encoding="utf-8-sig")

    if context.pairing_report_path.exists():
        pairing_report = pd.read_csv(context.pairing_report_path)
    else:
        pairing_report = pd.DataFrame()
    window_manifest, paired_window_manifest = build_window_manifests(context, preprocessed, pairing_report)
    return {
        "preprocessed_index": preprocessed_path,
        "window_manifest": window_manifest,
        "paired_window_manifest": paired_window_manifest,
    }


def build_pseudo_targets(preprocessed: pd.DataFrame, pseudo_dir: Path) -> dict[str, str]:
    group_to_path: dict[str, str] = {}
    for group_name, group in preprocessed.groupby("pseudo_target_group"):
        if len(group) < 2:
            continue
        signals = []
        lengths = []
        for cache_path in group["cache_path"]:
            signal = np.load(cache_path)["signal"]
            signals.append(signal)
            lengths.append(signal.shape[1])
        min_length = min(lengths)
        stacked = np.stack([signal[:, :min_length] for signal in signals], axis=0)
        averaged = stacked.mean(axis=0).astype(np.float32)
        target_path = pseudo_dir / f"{_stable_token(group_name)}.npz"
        np.savez_compressed(target_path, signal=averaged)
        group_to_path[group_name] = str(target_path)
    return group_to_path


def _window_starts(num_samples: int, window_size: int, stride: int) -> list[int]:
    if num_samples <= window_size:
        return [0]
    starts = list(range(0, num_samples - window_size + 1, stride))
    if starts[-1] != num_samples - window_size:
        starts.append(num_samples - window_size)
    return starts


def build_window_manifests(
    context: ProjectContext,
    preprocessed: pd.DataFrame,
    pairing_report: pd.DataFrame,
) -> tuple[Path, Path]:
    cfg = context.config["preprocessing"]
    window_size = int(cfg["window_size"])
    stride = int(cfg["stride"])
    window_rows: list[dict[str, Any]] = []

    for _, row in preprocessed.iterrows():
        available_length = int(row["num_samples"])
        pseudo_target_path = row.get("pseudo_target_path")
        if isinstance(pseudo_target_path, str) and pseudo_target_path:
            target_signal = np.load(pseudo_target_path)["signal"]
            available_length = min(available_length, int(target_signal.shape[1]))
        starts = _window_starts(available_length, window_size, stride)
        effective_size = min(window_size, available_length)
        for index, start in enumerate(starts):
            end = start + effective_size
            window_rows.append(
                {
                    "window_id": f"{row['raw_file_id']}__{index:04d}",
                    "raw_file_id": row["raw_file_id"],
                    "cache_path": row["cache_path"],
                    "pseudo_target_path": pseudo_target_path,
                    "start": start,
                    "end": end,
                    "position_id": row["position_id"],
                    "orientation": row["orientation"],
                    "pair_key": row["pair_key"],
                    "defect_unit_id": row["defect_unit_id"],
                }
            )

    window_manifest = pd.DataFrame(window_rows)
    window_manifest_path = context.reports_dir / "window_index.csv"
    window_manifest.to_csv(window_manifest_path, index=False, encoding="utf-8-sig")

    paired_rows: list[dict[str, Any]] = []
    if not pairing_report.empty:
        cache_lookup = preprocessed.set_index("raw_file_id")[["cache_path", "position_id"]].to_dict("index")
        paired = pairing_report[pairing_report["status"] == "paired"]
        for _, row in paired.iterrows():
            axial_ids = str(row["axial_raw_file_ids"]).split("|")
            radial_ids = str(row["radial_raw_file_ids"]).split("|")
            if len(axial_ids) != 1 or len(radial_ids) != 1:
                continue
            axial_id = axial_ids[0]
            radial_id = radial_ids[0]
            if axial_id not in cache_lookup or radial_id not in cache_lookup:
                continue
            axial_signal = np.load(cache_lookup[axial_id]["cache_path"])["signal"]
            radial_signal = np.load(cache_lookup[radial_id]["cache_path"])["signal"]
            min_length = min(axial_signal.shape[1], radial_signal.shape[1])
            starts = _window_starts(min_length, window_size, stride)
            effective_size = min(window_size, min_length)
            for index, start in enumerate(starts):
                end = start + effective_size
                paired_rows.append(
                    {
                        "paired_window_id": f"{row['pair_key']}__{index:04d}",
                        "pair_key": row["pair_key"],
                        "position_id": row["position_id"],
                        "axial_raw_file_id": axial_id,
                        "radial_raw_file_id": radial_id,
                        "axial_cache_path": cache_lookup[axial_id]["cache_path"],
                        "radial_cache_path": cache_lookup[radial_id]["cache_path"],
                        "start": start,
                        "end": end,
                    }
                )

    paired_manifest = pd.DataFrame(paired_rows)
    paired_manifest_path = context.reports_dir / "paired_window_index.csv"
    paired_manifest.to_csv(paired_manifest_path, index=False, encoding="utf-8-sig")
    return window_manifest_path, paired_manifest_path


def _load_npz_signal(cache_path: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if cache_path not in cache:
        cache[cache_path] = np.load(cache_path)["signal"].astype(np.float32)
    return cache[cache_path]


class SignalWindowDataset(Dataset):
    def __init__(self, window_frame: pd.DataFrame, use_pseudo_target: bool = True) -> None:
        self.window_frame = window_frame.reset_index(drop=True)
        self.use_pseudo_target = use_pseudo_target
        self.signal_cache: dict[str, np.ndarray] = {}
        self.target_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.window_frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.window_frame.iloc[index]
        signal = _load_npz_signal(row["cache_path"], self.signal_cache)
        start, end = int(row["start"]), int(row["end"])
        input_window = signal[:, start:end]
        target_window = input_window.copy()
        pseudo_target_path = row.get("pseudo_target_path")
        if self.use_pseudo_target and isinstance(pseudo_target_path, str) and pseudo_target_path:
            target_signal = _load_npz_signal(pseudo_target_path, self.target_cache)
            end = min(end, target_signal.shape[1])
            target_window = target_signal[:, start:end]
            input_window = input_window[:, : target_window.shape[1]]

        return {
            "inputs": torch.from_numpy(input_window),
            "targets": torch.from_numpy(target_window),
            "position_id": torch.tensor(int(row["position_id"]) - 1, dtype=torch.long),
            "pair_key": row.get("pair_key"),
            "raw_file_id": row["raw_file_id"],
        }


class PairedWindowDataset(Dataset):
    def __init__(self, paired_window_frame: pd.DataFrame) -> None:
        self.paired_window_frame = paired_window_frame.reset_index(drop=True)
        self.cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.paired_window_frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.paired_window_frame.iloc[index]
        axial = _load_npz_signal(row["axial_cache_path"], self.cache)
        radial = _load_npz_signal(row["radial_cache_path"], self.cache)
        start, end = int(row["start"]), int(row["end"])
        axial_window = axial[:, start:end]
        radial_window = radial[:, start:end]
        min_length = min(axial_window.shape[1], radial_window.shape[1])
        axial_window = axial_window[:, :min_length]
        radial_window = radial_window[:, :min_length]
        return {
            "axial": torch.from_numpy(axial_window),
            "radial": torch.from_numpy(radial_window),
            "position_id": torch.tensor(int(row["position_id"]) - 1, dtype=torch.long),
            "pair_key": row["pair_key"],
        }


class LabeledSignalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, target_columns: list[str], downsample_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.target_columns = target_columns
        self.downsample_length = downsample_length
        self.cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        signal = _load_npz_signal(row["cache_path"], self.cache)
        if signal.shape[1] > self.downsample_length:
            indices = np.linspace(0, signal.shape[1] - 1, self.downsample_length).astype(np.int64)
            signal = signal[:, indices]
        targets = row[self.target_columns].to_numpy(dtype=np.float32)
        return {
            "inputs": torch.from_numpy(signal),
            "targets": torch.from_numpy(targets),
            "raw_file_id": row["raw_file_id"],
        }
