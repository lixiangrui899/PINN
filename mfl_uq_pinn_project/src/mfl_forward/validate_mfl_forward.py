"""
Validation script for axisymmetric MFL forward PINN.

轴对称漏磁正问题硬边界 PINN 的验证脚本：使用简化磁偶极子真值进行评估。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from base_pinn.hard_boundary_pinn import ensure_dir
from data_processing.dipole_simulation import (
    effective_moment,
    generate_bz_signal_line,
    generate_potential_grid,
)
from mfl_forward.mfl_forward_pinn import build_mfl_pinn
from mfl_forward.mfl_geometry import GeometryConfig

dde.backend.set_default_backend("pytorch")


def set_seeds(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducibility.

    固定随机种子，保证结果可复现。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dde.config.set_random_seed(seed)


def prepare_anchors(cfg: GeometryConfig):
    """
    Build anchor points using simplified dipole scalar potential.

    使用简化磁偶极子标量磁位生成锚点，帮助 PINN 收敛到期望解。
    """

    moment = effective_moment(cfg.defect_depth, cfg.defect_length, cfg.defect_width)
    rr, zz, phi = generate_potential_grid(
        (0.0, cfg.r_max),
        (cfg.z_min, cfg.z_max),
        num_r=40,
        num_z=80,
        moment=moment,
    )
    anchors = np.stack([rr.ravel(), zz.ravel()], axis=1)
    anchor_values = phi.reshape(-1, 1)
    return anchors, anchor_values, rr, zz, phi


def plot_losses(history: dde.training.TrainingHistory, save_path: Path) -> None:
    """
    Plot training loss curve.

    绘制训练损失曲线。
    """

    losses = np.array(history.loss_train)
    total_loss = losses.sum(axis=1)
    domain_loss = losses[:, 0]
    boundary_loss = losses[:, 1] if losses.shape[1] > 1 else np.zeros_like(domain_loss)

    plt.figure(figsize=(6, 4))
    plt.semilogy(domain_loss, label="Domain loss")
    plt.semilogy(boundary_loss, label="Boundary/anchor loss")
    plt.semilogy(total_loss, label="Total loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_heatmaps(
    rr: np.ndarray,
    zz: np.ndarray,
    phi_true: np.ndarray,
    phi_pred: np.ndarray,
    save_path: Path,
) -> None:
    """
    Plot side-by-side heatmaps for true vs predicted scalar potential.

    绘制真值与预测的标量磁位热力图对比。
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].contourf(zz, rr, phi_true, levels=50, cmap="viridis")
    axes[0].set_title("True φ")
    axes[0].set_xlabel("z (m)")
    axes[0].set_ylabel("r (m)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(zz, rr, phi_pred, levels=50, cmap="viridis")
    axes[1].set_title("PINN φ")
    axes[1].set_xlabel("z (m)")
    axes[1].set_ylabel("r (m)")
    fig.colorbar(im1, ax=axes[1])

    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_metrics(
    pinn, rr: np.ndarray, zz: np.ndarray, phi_true: np.ndarray, cfg: GeometryConfig
):
    """
    Evaluate MSE, max absolute error, and boundary residual.

    计算 MSE、最大绝对误差与边界残差。
    """

    x_grid = np.stack([rr.ravel(), zz.ravel()], axis=1)
    phi_pred = pinn.predict(x_grid).reshape(rr.shape)

    mse = float(np.mean((phi_pred - phi_true) ** 2))
    max_abs = float(np.max(np.abs(phi_pred - phi_true)))

    z_line = np.linspace(cfg.z_min, cfg.z_max, 100)
    r_line = np.full_like(z_line, cfg.r_max)
    boundary_vals = pinn.predict(np.stack([r_line, z_line], axis=1))
    boundary_res = float(np.max(np.abs(boundary_vals)))
    return mse, max_abs, boundary_res, phi_pred


def main() -> None:
    """
    End-to-end pipeline for MFL forward validation.

    端到端流程：构建锚点、训练 PINN、评估并绘图。
    """

    set_seeds(42)
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results" / "mfl_forward"
    ensure_dir(results_dir)

    cfg = GeometryConfig()
    anchors, anchor_values, rr, zz, phi_true = prepare_anchors(cfg)

    pinn, ckpt_path = build_mfl_pinn(
        workdir=root, cfg=cfg, anchors=anchors, anchor_values=anchor_values
    )
    history = pinn.train(
        iterations=20000, checkpoint_path=ckpt_path, display_every=2000
    )

    mse, max_abs, boundary_res, phi_pred = evaluate_metrics(
        pinn, rr, zz, phi_true, cfg
    )

    print("=== Validation Metrics (MFL Forward) ===")
    print(f"Boundary residual max: {boundary_res:.3e}")
    print(f"MSE vs dipole truth: {mse:.3e}")
    print(f"Max abs error vs dipole truth: {max_abs:.3e}")

    plot_losses(history, results_dir / "loss_curve.png")
    plot_heatmaps(rr, zz, phi_true, phi_pred, results_dir / "phi_heatmap.png")

    assert boundary_res <= 1e-5, "Boundary residual exceeds 1e-5"
    assert mse <= 5e-3, "MSE exceeds 5e-3"

    print("All acceptance criteria passed for MFL forward task.")


if __name__ == "__main__":
    main()
