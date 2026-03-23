"""
Simplified magnetic dipole simulation utilities for MFL.

简化磁偶极子模型仿真工具：生成标量磁位与漏磁 Bz 信号。
"""

from __future__ import annotations

import numpy as np

MU0 = 4 * np.pi * 1e-7  # Vacuum permeability / 真空磁导率


def effective_moment(depth: float, length: float, width: float) -> float:
    """
    Compute an effective dipole moment proportional to defect volume.

    以缺陷体积近似计算等效偶极矩，便于快速生成信号。
    """

    volume = depth * length * width
    return 1e5 * volume  # scale factor tuned for signal magnitude


def scalar_potential(
    r: np.ndarray, z: np.ndarray, moment: float = 1.0, z0: float = 0.0
) -> np.ndarray:
    """
    Compute scalar magnetic potential of a z-directed dipole.

    计算轴向偶极子在柱坐标系下的标量磁位 φ = m (z-z0) / (4π μ0 (r²+(z-z0)²)^{3/2})。
    """

    rz2 = r**2 + (z - z0) ** 2
    return moment * (z - z0) / (4 * np.pi * MU0 * np.power(rz2, 1.5) + 1e-12)


def bz_dipole(
    r: np.ndarray, z: np.ndarray, moment: float = 1.0, z0: float = 0.0
) -> np.ndarray:
    """
    Compute Bz component of a z-directed dipole in cylindrical coordinates.

    计算轴向偶极子的 Bz 分量：Bz = μ0 m (2(z-z0)^2 - r^2) / (4π (r^2+(z-z0)^2)^{5/2})。
    """

    rz2 = r**2 + (z - z0) ** 2
    numerator = MU0 * moment * (2 * (z - z0) ** 2 - r**2)
    denominator = 4 * np.pi * np.power(rz2, 2.5) + 1e-12
    return numerator / denominator


def generate_potential_grid(
    r_range: tuple[float, float],
    z_range: tuple[float, float],
    num_r: int,
    num_z: int,
    moment: float,
    z0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate meshgrid (r, z) and scalar potential values for visualization or anchors.

    生成 (r, z) 网格以及对应的标量磁位，可用于验证或锚点约束。
    """

    r = np.linspace(r_range[0], r_range[1], num_r)
    z = np.linspace(z_range[0], z_range[1], num_z)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    phi = scalar_potential(rr, zz, moment=moment, z0=z0)
    return rr, zz, phi


def generate_bz_signal_line(
    r_value: float,
    z_range: tuple[float, float],
    num_z: int,
    moment: float,
    z0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Bz signal along a line r = const.

    在固定径向 r=r_value 上生成 Bz 轴向分布，用于漏磁检测信号。
    """

    z = np.linspace(z_range[0], z_range[1], num_z)
    r = np.full_like(z, r_value)
    bz = bz_dipole(r, z, moment=moment, z0=z0)
    return z, bz
