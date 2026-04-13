from __future__ import annotations

import random
from typing import Any

import pandas as pd

from uq_pinn_mfl.config import ProjectContext


def build_lopo_schedule(position_ids: list[int], preferred_validation_position: int | None = None) -> list[dict[str, int]]:
    unique_positions = sorted({int(position_id) for position_id in position_ids})
    schedule: list[dict[str, int]] = []
    for index, holdout_position in enumerate(unique_positions):
        remaining = [position for position in unique_positions if position != holdout_position]
        if not remaining:
            continue
        if preferred_validation_position is not None and preferred_validation_position in remaining:
            validation_position = preferred_validation_position
        else:
            validation_position = remaining[index % len(remaining)]
        schedule.append(
            {
                "holdout_position": holdout_position,
                "validation_position": validation_position,
            }
        )
    return schedule


def leave_one_position_out(dataset_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unique_positions = sorted(dataset_index["position_id"].dropna().unique().tolist())
    for fold_position in unique_positions:
        for _, row in dataset_index.iterrows():
            subset = "test" if row["position_id"] == fold_position else "train"
            rows.append(
                {
                    "strategy": "leave_one_position_out",
                    "fold": f"holdout_position_{fold_position}",
                    "subset": subset,
                    "raw_file_id": row["raw_file_id"],
                    "pair_key": row.get("pair_key"),
                    "position_id": row.get("position_id"),
                }
            )
    return pd.DataFrame(rows)


def grouped_kfold_by_defect_unit(dataset_index: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unique_units = dataset_index["defect_unit_id"].dropna().unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(unique_units)
    folds = [unique_units[index::n_splits] for index in range(n_splits)]

    for fold_idx, fold_units in enumerate(folds):
        fold_name = f"group_kfold_{fold_idx + 1}"
        fold_units_set = set(fold_units)
        for _, row in dataset_index.iterrows():
            subset = "test" if row["defect_unit_id"] in fold_units_set else "train"
            rows.append(
                {
                    "strategy": "grouped_kfold_by_defect_unit_id",
                    "fold": fold_name,
                    "subset": subset,
                    "raw_file_id": row["raw_file_id"],
                    "pair_key": row.get("pair_key"),
                    "position_id": row.get("position_id"),
                }
            )
    return pd.DataFrame(rows)


def strict_paired_split(pairing_report: pd.DataFrame, seed: int, val_fraction: float, test_fraction: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paired = pairing_report[pairing_report["status"] == "paired"].copy()
    pair_keys = paired["pair_key"].dropna().tolist()
    rng = random.Random(seed)
    rng.shuffle(pair_keys)

    total = len(pair_keys)
    test_count = max(1, int(total * test_fraction)) if total else 0
    val_count = max(1, int(total * val_fraction)) if total > 2 else 0
    test_keys = set(pair_keys[:test_count])
    val_keys = set(pair_keys[test_count : test_count + val_count])
    train_keys = set(pair_keys[test_count + val_count :])

    for _, row in pairing_report.iterrows():
        pair_key = row["pair_key"]
        if pair_key in test_keys:
            subset = "test"
        elif pair_key in val_keys:
            subset = "val"
        elif pair_key in train_keys:
            subset = "train"
        else:
            subset = "excluded"
        rows.append(
            {
                "strategy": "strict_paired_split",
                "fold": "default",
                "subset": subset,
                "pair_key": pair_key,
                "position_id": row.get("position_id"),
                "axial_raw_file_ids": row.get("axial_raw_file_ids"),
                "radial_raw_file_ids": row.get("radial_raw_file_ids"),
            }
        )
    return pd.DataFrame(rows)


def write_split_manifests(context: ProjectContext, dataset_index: pd.DataFrame, pairing_report: pd.DataFrame) -> dict[str, str]:
    cfg = context.config["splits"]
    lopo = leave_one_position_out(dataset_index)
    group_kfold = grouped_kfold_by_defect_unit(dataset_index, int(cfg["n_splits"]), int(cfg["seed"]))
    paired = strict_paired_split(
        pairing_report,
        seed=int(cfg["seed"]),
        val_fraction=float(cfg["val_fraction"]),
        test_fraction=float(cfg["test_fraction"]),
    )

    outputs = {
        "leave_one_position_out": context.manifests_dir / "leave_one_position_out.csv",
        "grouped_kfold_by_defect_unit_id": context.manifests_dir / "grouped_kfold_by_defect_unit_id.csv",
        "strict_paired_split": context.manifests_dir / "strict_paired_split.csv",
    }
    lopo.to_csv(outputs["leave_one_position_out"], index=False, encoding="utf-8-sig")
    group_kfold.to_csv(outputs["grouped_kfold_by_defect_unit_id"], index=False, encoding="utf-8-sig")
    paired.to_csv(outputs["strict_paired_split"], index=False, encoding="utf-8-sig")
    return {name: str(path) for name, path in outputs.items()}
