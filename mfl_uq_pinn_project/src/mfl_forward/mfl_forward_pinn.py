"""
MFL forward PINN model builder built on the hard-boundary base class.

基于通用硬边界 PINN 的漏磁正问题模型构建器。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import deepxde as dde
import numpy as np

from base_pinn.hard_boundary_pinn import HardBoundaryPINN, NetworkConfig, ensure_dir
from mfl_forward.mfl_geometry import GeometryConfig, build_rectangle
from mfl_forward.mfl_pde import axisymmetric_mfl_pde, output_transform_builder


def boundary_outer(x: np.ndarray, _: int, cfg: GeometryConfig) -> np.ndarray:
    """
    Identify outer detection boundary r = r_max.

    判定外侧检测区域边界 r = r_max。
    """

    return np.isclose(x[0], cfg.r_max)


def build_mfl_pinn(
    workdir: Path,
    cfg: GeometryConfig,
    anchors: Optional[np.ndarray] = None,
    anchor_values: Optional[np.ndarray] = None,
) -> Tuple[HardBoundaryPINN, Path]:
    """
    Assemble the MFL forward PINN with hard boundary constraints.

    构建漏磁正问题 PINN：包含外边界 Dirichlet 硬约束和轴对称 Neumann 硬约束。
    """

    geom = build_rectangle(cfg)
    bc_outer = dde.icbc.DirichletBC(
        geom, lambda _: 0.0, lambda x, on_boundary: on_boundary and boundary_outer(x, 0, cfg)
    )

    net_config = NetworkConfig(
        layer_sizes=[2, 128, 128, 128, 128, 128, 1],
        activation="tanh",
        initializer="Glorot normal",
    )
    model_dir = workdir / "models" / "mfl_forward"
    ensure_dir(model_dir)

    pinn = HardBoundaryPINN(
        geometry=geom,
        pde=axisymmetric_mfl_pde(cfg),
        boundary_conds=[bc_outer],
        output_transform=output_transform_builder(cfg),
        net_config=net_config,
        model_dir=model_dir,
    )
    pinn.build_data(
        num_domain=2000,
        num_boundary=200,
        anchors=anchors,
        anchor_values=anchor_values,
    )
    pinn.build_model()
    pinn.compile_model(lr=1e-4)

    ckpt_path = model_dir / "model.ckpt"
    pinn.load_checkpoint(ckpt_path)
    return pinn, ckpt_path

