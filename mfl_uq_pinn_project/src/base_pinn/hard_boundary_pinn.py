"""
Hard-boundary PINN framework built with DeepXDE.

硬边界 PINN 通用框架，基于 DeepXDE 封装训练、恢复与可视化接口。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import deepxde as dde
import numpy as np


@dataclass
class NetworkConfig:
    """
    Network hyperparameter container.

    网络超参数容器，统一管理层数、激活函数与初始化方式。
    """

    layer_sizes: Sequence[int]
    activation: str = "tanh"
    initializer: str = "Glorot normal"


class HardBoundaryPINN:
    """
    General hard-boundary constrained PINN wrapper.

    通用硬边界 PINN 封装：通过 output_transform 强制满足边界条件，
    并提供训练、断点续训与结果保存接口。
    """

    def __init__(
        self,
        geometry: dde.geometry.Geometry,
        pde: Callable,
        boundary_conds: Optional[List[dde.icbc.BC]] = None,
        output_transform: Optional[Callable] = None,
        net_config: Optional[NetworkConfig] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the PINN model wrapper.

        初始化 PINN 封装，构造数据对象但推迟模型构建。

        Args:
            geometry: DeepXDE geometry object defining the domain.
            pde: PDE residual function f(x, y) -> residual.
            boundary_conds: Optional list of BCs; 可选的边界条件列表，可用于监控残差。
            output_transform: Output transform enforcing hard BCs; 输出变换确保硬边界。
            net_config: NetworkConfig defining architecture; 网络结构配置。
            model_dir: Directory to store checkpoints; 模型保存目录。
        """

        self.geometry = geometry
        self.pde = pde
        self.boundary_conds = boundary_conds or []
        self.output_transform = output_transform
        self.net_config = net_config or NetworkConfig(layer_sizes=(2, 50, 50, 1))
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data: Optional[dde.data.PDE] = None
        self.model: Optional[dde.Model] = None
        self.history: Optional[dde.training.TrainingHistory] = None

    def build_data(
        self,
        num_domain: int = 1000,
        num_boundary: int = 200,
        anchors: Optional[np.ndarray] = None,
        anchor_values: Optional[np.ndarray] = None,
    ) -> None:
        """
        Construct DeepXDE PDE data object.

        构建 DeepXDE 的 PDE 数据集，包含内部采样和边界采样。

        Args:
            num_domain: Interior collocation points; 内部采样点数量。
            num_boundary: Boundary points; 边界采样点数量。
            anchors: Optional supervised anchor coordinates; 可选锚点坐标，用于加入真值约束。
            anchor_values: Anchor target values; 锚点目标真值。
        """

        self.data = dde.data.PDE(
            self.geometry,
            self.pde,
            self.boundary_conds,
            num_domain=num_domain,
            num_boundary=num_boundary,
            solution=None,
            anchors=anchors,
            anchor_values=anchor_values,
        )

    def build_model(self) -> None:
        """
        Build the DeepXDE model with output transform.

        构造 DeepXDE 模型，并在网络输出端口施加硬边界输出变换。
        """

        if self.data is None:
            raise ValueError("Data must be built before building the model.")

        net = dde.nn.FNN(
            layer_sizes=self.net_config.layer_sizes,
            activation=self.net_config.activation,
            initializer=self.net_config.initializer,
        )
        if self.output_transform is not None:
            net.apply_output_transform(self.output_transform)

        self.model = dde.Model(self.data, net)

    def compile_model(self, lr: float = 1e-4) -> None:
        """
        Compile the model with Adam optimizer.

        使用 Adam 优化器编译模型，学习率可配置。
        """

        if self.model is None:
            raise ValueError("Model must be built before compilation.")
        self.model.compile("adam", lr=lr)

    def train(
        self,
        iterations: int,
        checkpoint_path: Optional[Path] = None,
        display_every: int = 1000,
    ) -> dde.training.TrainingHistory:
        """
        Train the model with checkpointing.

        训练模型并保存检查点，支持断点续训。

        Args:
            iterations: Number of training iterations; 训练迭代次数。
            checkpoint_path: Path to save weights; 权重保存路径。
            display_every: Logging frequency; 日志打印频率。
        """

        if self.model is None:
            raise ValueError("Model must be compiled before training.")

        ckpt_path = checkpoint_path or (self.model_dir / "model.ckpt")
        callbacks = [
            dde.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path), save_better_only=True, verbose=0
            )
        ]
        self.history, _ = self.model.train(
            iterations=iterations, display_every=display_every, callbacks=callbacks
        )
        return self.history

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> bool:
        """
        Load weights if a checkpoint exists.

        检查并加载已有检查点，用于断点续训。

        Returns:
            bool: True if checkpoint loaded successfully.
        """

        if self.model is None:
            raise ValueError("Model must be built before loading checkpoints.")

        ckpt_path = checkpoint_path or (self.model_dir / "model.ckpt")
        if ckpt_path.exists():
            self.model.restore(str(ckpt_path), verbose=1)
            return True
        return False

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run model inference.

        执行模型推理，返回预测结果。
        """

        if self.model is None:
            raise ValueError("Model must be compiled and trained before prediction.")
        return self.model.predict(x)

    def evaluate_loss(
        self, x: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute MSE and max absolute error against ground truth.

        计算与真值的均方误差和最大绝对误差。
        """

        y_pred = self.predict(x)
        mse = float(np.mean((y_pred - y_true) ** 2))
        max_abs = float(np.max(np.abs(y_pred - y_true)))
        return mse, max_abs


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not exist.

    若目录不存在则创建，确保保存路径可用。
    """

    os.makedirs(path, exist_ok=True)
