"""
Axisymmetric MFL PDE and material properties.

轴对称漏磁正问题的控制方程与材料磁导率定义。
"""

from __future__ import annotations

import deepxde as dde
import torch

from mfl_forward.mfl_geometry import GeometryConfig

MU0 = 4 * torch.pi * 1e-7  # Vacuum permeability / 真空磁导率
MU_R_IRON = 1000.0  # Relative permeability for ferromagnetic wall / 铁磁材料相对磁导率


def material_mu(r: torch.Tensor, z: torch.Tensor, cfg: GeometryConfig) -> torch.Tensor:
    """
    Piecewise magnetic permeability function.

    分段磁导率：空气区 μ0，管壁区 μr*μ0，缺陷区复原为 μ0。
    """

    mu_air = MU0
    mu_wall = MU0 * MU_R_IRON

    wall_mask = (r >= cfg.r_in) & (r <= cfg.r_out)
    defect_mask = wall_mask & (r <= cfg.r_in + cfg.defect_depth) & (
        torch.abs(z) <= cfg.defect_length / 2
    )

    mu = torch.where(wall_mask, mu_wall, mu_air)
    mu = torch.where(defect_mask, mu_air, mu)
    return mu


def axisymmetric_mfl_pde(
    cfg: GeometryConfig,
):
    """
    Build PDE residual function for DeepXDE.

    构造 DeepXDE 所需的 PDE 残差函数： (1/r) ∂/∂r (r μ ∂φ/∂r) + ∂/∂z(μ ∂φ/∂z) = 0。
    """

    def pde(x, y):
        r = x[:, 0:1]
        z = x[:, 1:2]
        r_safe = torch.clamp(r, min=1e-5)

        mu_val = material_mu(r, z, cfg)

        dphi_dr = dde.grad.jacobian(y, x, i=0, j=0)
        dphi_dz = dde.grad.jacobian(y, x, i=0, j=1)

        radial_flux = mu_val * r_safe * dphi_dr
        axial_flux = mu_val * dphi_dz

        term_r = (1.0 / r_safe) * dde.grad.jacobian(radial_flux, x, i=0, j=0)
        term_z = dde.grad.jacobian(axial_flux, x, i=0, j=1)
        return term_r + term_z

    return pde


def output_transform_builder(cfg: GeometryConfig):
    """
    Create output transform enforcing φ=0 at r=r_max and ∂φ/∂r|_{r=0}=0.

    构造输出变换：在外边界 r=r_max 上满足 Dirichlet 零值，在轴向 r=0 处通过 r^2 因子确保径向导数为零。
    """

    def transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = x[:, 0:1]
        return r**2 * (1 - r / cfg.r_max) * y

    return transform
