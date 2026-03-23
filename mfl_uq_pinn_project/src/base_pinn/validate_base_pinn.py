"""
Validation script for hard-boundary PINN on 1D Poisson equation.

一维泊松方程的硬边界 PINN 验证脚本：自动训练、评估并生成可视化结果。
"""

from __future__ import annotations

import random
from pathlib import Path
import sys

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running as a script by adding project src to sys.path
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from base_pinn.hard_boundary_pinn import HardBoundaryPINN, NetworkConfig, ensure_dir

# 强制使用 PyTorch 后端 / Force PyTorch backend
dde.backend.set_default_backend("pytorch")


def set_seeds(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducibility.

    固定随机种子，确保结果可复现。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dde.config.set_random_seed(seed)


def analytic_solution(x: np.ndarray) -> np.ndarray:
    """
    Analytical solution u(x) = x(1 - x^2).

    泊松方程解析解，用于精度验证。
    """

    return x * (1 - x**2)


def pde_poisson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Define 1D Poisson PDE residual: d2u/dx2 + 6x = 0.

    一维泊松方程残差：二阶导数加 6x 等于 0，作为 PINN 损失。
    """

    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    d2y_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    return d2y_xx + 6 * x[:, 0:1]


def boundary_left(x: np.ndarray, _: int) -> np.ndarray:
    """
    Identify left boundary x=0.

    判定左边界 x=0。
    """

    return np.isclose(x[0], 0.0)


def boundary_right(x: np.ndarray, _: int) -> np.ndarray:
    """
    Identify right boundary x=1.

    判定右边界 x=1。
    """

    return np.isclose(x[0], 1.0)


def output_transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Output transform enforcing Dirichlet zero at x=0 and x=1.

    通过输出变换 û = x(1-x)·N(x) 强制满足 u(0)=u(1)=0。
    """

    return x[:, 0:1] * (1 - x[:, 0:1]) * y


def build_pinn(
    workdir: Path, num_domain: int = 100, num_boundary: int = 20
) -> tuple[HardBoundaryPINN, Path]:
    """
    Construct the hard-boundary PINN instance for the Poisson test case.

    创建针对泊松方程的硬边界 PINN，并返回模型对象与检查点路径。
    """

    geom = dde.geometry.Interval(0.0, 1.0)
    bc_left = dde.icbc.DirichletBC(geom, lambda _: 0.0, boundary_left)
    bc_right = dde.icbc.DirichletBC(geom, lambda _: 0.0, boundary_right)

    net_config = NetworkConfig(layer_sizes=[1, 64, 64, 64, 64, 1])
    model_dir = workdir / "models" / "base_pinn"
    ensure_dir(model_dir)

    pinn = HardBoundaryPINN(
        geometry=geom,
        pde=pde_poisson,
        boundary_conds=[bc_left, bc_right],
        output_transform=output_transform,
        net_config=net_config,
        model_dir=model_dir,
    )
    pinn.build_data(num_domain=num_domain, num_boundary=num_boundary)
    pinn.build_model()
    pinn.compile_model(lr=1e-4)

    ckpt_path = model_dir / "model.ckpt"
    pinn.load_checkpoint(ckpt_path)
    return pinn, ckpt_path


def evaluate_boundary_residual(pinn: HardBoundaryPINN) -> np.ndarray:
    """
    Evaluate boundary residual |u(x)| at x=0 and x=1.

    计算边界残差：在 x=0 与 x=1 处预测值的绝对值。
    """

    x_bc = np.array([[0.0], [1.0]])
    y_bc = pinn.predict(x_bc)
    return np.abs(y_bc).reshape(-1)


def plot_losses(history: dde.training.TrainingHistory, save_path: Path) -> None:
    """
    Plot domain, boundary, and total losses.

    绘制域内损失、边界损失与总损失曲线。
    """

    losses = np.array(history.loss_train)
    total_loss = losses.sum(axis=1)
    domain_loss = losses[:, 0]
    boundary_loss = losses[:, 1] if losses.shape[1] > 1 else np.zeros_like(domain_loss)

    plt.figure(figsize=(6, 4))
    plt.semilogy(domain_loss, label="Domain loss")
    plt.semilogy(boundary_loss, label="Boundary loss")
    plt.semilogy(total_loss, label="Total loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prediction(
    x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path
) -> None:
    """
    Plot analytical vs predicted solutions.

    绘制解析解与 PINN 预测对比曲线。
    """

    plt.figure(figsize=(6, 4))
    plt.plot(x, y_true, label="Analytical", linewidth=2)
    plt.plot(x, y_pred, "--", label="PINN", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_boundary_residual(residuals: np.ndarray, save_path: Path) -> None:
    """
    Plot boundary residuals as a small line chart.

    以折线形式展示两端点的边界残差。
    """

    plt.figure(figsize=(5, 3))
    plt.plot([0, 1], residuals, marker="o")
    plt.xlabel("Boundary x")
    plt.ylabel("|u(x)| residual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    """
    End-to-end training and validation pipeline.

    端到端流程：训练、误差评估、绘图与验收判断。
    """

    set_seeds(42)
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results" / "base_pinn"
    ensure_dir(results_dir)

    pinn, ckpt_path = build_pinn(root)
    history = pinn.train(
        iterations=10000, checkpoint_path=ckpt_path, display_every=1000
    )

    x_test = np.linspace(0, 1, 200)[:, None]
    y_true = analytic_solution(x_test)
    y_pred = pinn.predict(x_test)

    mse, max_abs = pinn.evaluate_loss(x_test, y_true)
    boundary_res = evaluate_boundary_residual(pinn)
    max_boundary_res = float(np.max(boundary_res))

    print("=== Validation Metrics ===")
    print(f"Boundary residual max: {max_boundary_res:.3e}")
    print(f"MSE vs analytical: {mse:.3e}")
    print(f"Max abs error vs analytical: {max_abs:.3e}")

    plot_losses(history, results_dir / "loss_curve.png")
    plot_prediction(x_test, y_true, y_pred, results_dir / "prediction_vs_truth.png")
    plot_boundary_residual(boundary_res, results_dir / "boundary_residual.png")

    assert max_boundary_res <= 1e-6, "Boundary residual exceeds 1e-6"
    assert mse <= 1e-4, "MSE exceeds 1e-4"
    assert max_abs <= 1e-2, "Max absolute error exceeds 1e-2"

    print("All acceptance criteria passed.")


if __name__ == "__main__":
    main()
