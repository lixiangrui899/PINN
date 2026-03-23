"""
COMSOL data conversion helpers.

COMSOL 导出数据转换：提取检测区域 Bz，并按项目格式保存。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_comsol_table(path: Path) -> pd.DataFrame:
    """
    Load COMSOL txt/csv data into a DataFrame.

    读取 COMSOL 导出的 txt/csv 数据为 DataFrame。
    """

    return pd.read_csv(path, sep=None, engine="python")


def extract_detection_bz(
    df: pd.DataFrame, r_target: float, tol: float = 1e-4
) -> np.ndarray:
    """
    Extract Bz values along r = r_target within tolerance.

    按 r=r_target（允许微小偏差）提取检测线上的 Bz 信号。
    """

    mask = np.isclose(df["r"].values, r_target, atol=tol)
    subset = df.loc[mask].sort_values(by="z")
    return subset["z"].to_numpy(), subset["B_z"].to_numpy()


def convert_comsol_to_npz(
    input_path: Path,
    output_path: Path,
    defect_params: Dict[str, float],
    r_target: float,
) -> None:
    """
    Convert COMSOL table to standardized npz with z, Bz, and defect parameters.

    将 COMSOL 数据表转换为标准 npz，包含 z 轴坐标、Bz 信号与缺陷参数。
    """

    df = load_comsol_table(input_path)
    z, bz = extract_detection_bz(df, r_target=r_target)

    params = np.array(
        [
            [
                defect_params.get("h_def", 0.0),
                defect_params.get("l_def", 0.0),
                defect_params.get("w_def", 0.0),
            ]
        ]
    )

    np.savez_compressed(
        output_path,
        z=z,
        bz=bz,
        defect_params=params,
        metadata={"source": "COMSOL", "r_target": r_target},
    )

