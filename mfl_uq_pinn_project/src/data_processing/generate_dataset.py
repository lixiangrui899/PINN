"""
Dataset generation script for standard rectangular defects.

标准矩形缺陷磁偶极子仿真数据集生成与预处理脚本。
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from base_pinn.hard_boundary_pinn import ensure_dir
from data_processing.data_preprocessing import (
    minmax_normalize_params,
    split_dataset,
    zscore_normalize,
)
from data_processing.dipole_simulation import (
    effective_moment,
    generate_bz_signal_line,
)
from mfl_forward.mfl_geometry import GeometryConfig


def build_defect_grid():
    """
    Construct parameter grid covering required ranges.

    构造缺陷参数网格，覆盖深度/长度/宽度的指定范围。
    """

    depths = np.arange(0.001, 0.009, 0.001)  # 0.001~0.008
    lengths = np.arange(0.01, 0.051, 0.01)  # 0.01~0.05
    widths = np.arange(0.01, 0.051, 0.01)  # 0.01~0.05
    grid = []
    for h in depths:
        for l in lengths:
            for w in widths:
                grid.append((h, l, w))
    return grid


def simulate_dataset(cfg: GeometryConfig, num_z: int = 100):
    """
    Generate dipole Bz signals for all parameter combinations.

    针对所有缺陷组合生成 Bz 信号矩阵。
    """

    params_list = []
    signals = []
    z_axis = None

    for h, l, w in build_defect_grid():
        moment = effective_moment(h, l, w)
        z, bz = generate_bz_signal_line(
            r_value=cfg.r_out + cfg.lift_off,
            z_range=(cfg.z_min, cfg.z_max),
            num_z=num_z,
            moment=moment,
        )
        if z_axis is None:
            z_axis = z
        params_list.append([h, l, w])
        signals.append(bz)

    return np.array(signals), np.array(params_list), z_axis


def main() -> None:
    """
    End-to-end dataset generation and preprocessing.

    端到端数据生成与预处理：仿真、归一化、划分并保存。
    """

    cfg = GeometryConfig()
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    signals, params, z_axis = simulate_dataset(cfg, num_z=100)

    # Save raw (optional)
    np.savez_compressed(
        raw_dir / "dipole_raw_signals.npz",
        z=z_axis,
        signals=signals,
        params=params,
        metadata={"note": "raw dipole simulations"},
    )

    # Normalization
    signals_norm, z_stats = zscore_normalize(signals)
    min_vals = np.array([0.001, 0.01, 0.01])
    max_vals = np.array([0.008, 0.05, 0.05])
    params_norm, mm_stats = minmax_normalize_params(params, min_vals, max_vals)

    # Split dataset
    (train_x, train_p), (val_x, val_p), (test_x, test_p) = split_dataset(
        signals_norm, params_norm, train_ratio=0.7, val_ratio=0.2, seed=42
    )

    np.savez_compressed(
        processed_dir / "mfl_standard_defect_dataset.npz",
        train_bz=train_x,
        train_params=train_p,
        val_bz=val_x,
        val_params=val_p,
        test_bz=test_x,
        test_params=test_p,
        z_axis=z_axis,
        zscore_mean=z_stats.mean,
        zscore_std=z_stats.std,
        param_min=mm_stats.min_vals,
        param_max=mm_stats.max_vals,
        metadata={"seed": 42, "description": "200-sample dipole dataset"},
    )

    print("Dataset generated and saved to data/processed/mfl_standard_defect_dataset.npz")
    print(f"Train/Val/Test sizes: {len(train_x)}, {len(val_x)}, {len(test_x)}")


if __name__ == "__main__":
    main()
