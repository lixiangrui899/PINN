"""
Geometry definitions for axisymmetric MFL forward problem.

轴对称漏磁正问题的几何与尺寸参数封装。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import deepxde as dde


@dataclass
class GeometryConfig:
    """
    Geometry configuration for pipe, defect, and lift-off.

    几何参数配置：管道尺寸、缺陷尺寸与提离距离。
    """

    r_in: float = 0.1
    r_out: float = 0.11
    length: float = 0.5
    lift_off: float = 0.002
    defect_depth: float = 0.005
    defect_length: float = 0.03
    defect_width: float = 0.02  # width used for dataset compatibility

    @property
    def r_max(self) -> float:
        """Outer radial limit including lift-off / 外侧半径（含提离）"""

        return self.r_out + self.lift_off

    @property
    def z_min(self) -> float:
        """Lower axial bound / 轴向最小值"""

        return -self.length / 2

    @property
    def z_max(self) -> float:
        """Upper axial bound / 轴向最大值"""

        return self.length / 2


def build_rectangle(cfg: GeometryConfig) -> dde.geometry.Rectangle:
    """
    Build the global rectangular computational domain (r, z).

    构造包含空气区与管壁的矩形计算域。
    """

    return dde.geometry.Rectangle([0.0, cfg.z_min], [cfg.r_max, cfg.z_max])

