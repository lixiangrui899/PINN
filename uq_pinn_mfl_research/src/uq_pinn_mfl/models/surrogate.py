from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def first_derivative(signal: torch.Tensor) -> torch.Tensor:
    derivative = signal[..., 1:] - signal[..., :-1]
    return F.pad(derivative, (0, 1))


def second_derivative(signal: torch.Tensor) -> torch.Tensor:
    derivative = first_derivative(signal)
    second = derivative[..., 1:] - derivative[..., :-1]
    return F.pad(second, (0, 1))


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 8, hidden_channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class BlackBoxProxyNet(nn.Module):
    def __init__(self, in_channels: int = 8, num_positions: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, hidden_channels=32, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, in_channels, kernel_size=1),
        )
        self.position_head = nn.Linear(128, num_positions)
        self.confidence_head = nn.Linear(128, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        embedding = torch.cat([features.mean(dim=-1), features.std(dim=-1)], dim=1)
        return {
            "embedding": embedding,
            "reconstruction": self.decoder(features),
            "position_logits": self.position_head(embedding),
            "proxy_logvar": self.confidence_head(embedding),
        }


class BlackBoxRegressor(nn.Module):
    def __init__(self, in_channels: int = 8, target_dim: int = 4, use_uq: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        self.use_uq = use_uq
        self.encoder = ConvEncoder(in_channels=in_channels, hidden_channels=32, dropout=dropout)
        self.mean_head = nn.Linear(128, target_dim)
        self.logvar_head = nn.Linear(128, target_dim)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        embedding = torch.cat([features.mean(dim=-1), features.std(dim=-1)], dim=1)
        mean = self.mean_head(embedding)
        logvar = self.logvar_head(embedding) if self.use_uq else torch.zeros_like(mean)
        return {
            "embedding": embedding,
            "regression_mean": mean,
            "regression_logvar": logvar,
        }


class PhysicsGuidedSurrogate(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        num_positions: int = 5,
        target_dim: int = 4,
        use_uq: bool = True,
        use_discrepancy: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_uq = use_uq
        self.use_discrepancy = use_discrepancy
        self.encoder = ConvEncoder(in_channels=in_channels, hidden_channels=32, dropout=dropout)
        self.state_head = nn.Conv1d(64, 1, kernel_size=1)
        self.discrepancy_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, in_channels, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.randn(in_channels, 1) * 0.05)
        self.beta = nn.Parameter(torch.randn(in_channels, 1) * 0.05)
        self.position_head = nn.Linear(128, num_positions)
        self.reg_head = nn.Linear(128, target_dim)
        self.reg_logvar_head = nn.Linear(128, target_dim)
        self.proxy_logvar_head = nn.Linear(128, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        state = self.state_head(features)
        grad = first_derivative(state)
        physics_only = self.alpha.unsqueeze(0) * state + self.beta.unsqueeze(0) * grad
        discrepancy = self.discrepancy_net(state) if self.use_discrepancy else torch.zeros_like(physics_only)
        reconstruction = physics_only + discrepancy
        embedding = torch.cat([features.mean(dim=-1), features.std(dim=-1)], dim=1)
        outputs = {
            "embedding": embedding,
            "state": state,
            "physics_only_reconstruction": physics_only,
            "reconstruction": reconstruction,
            "discrepancy": discrepancy,
            "position_logits": self.position_head(embedding),
            "proxy_logvar": self.proxy_logvar_head(embedding),
            "regression_mean": self.reg_head(embedding),
        }
        outputs["regression_logvar"] = self.reg_logvar_head(embedding) if self.use_uq else torch.zeros_like(outputs["regression_mean"])
        return outputs


def compute_proxy_loss(
    outputs: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    position_targets: torch.Tensor,
    physics_weight: float,
    discrepancy_weight: float,
    position_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    data_loss = F.mse_loss(outputs["reconstruction"], inputs)
    physics_loss = F.mse_loss(outputs["physics_only_reconstruction"], inputs) + second_derivative(outputs["state"]).pow(2).mean()
    discrepancy = outputs["discrepancy"].pow(2).mean()
    position = F.cross_entropy(outputs["position_logits"], position_targets)
    total = data_loss + physics_weight * physics_loss + discrepancy_weight * discrepancy + position_weight * position
    metrics = {
        "data_loss": float(data_loss.detach().cpu()),
        "physics_loss": float(physics_loss.detach().cpu()),
        "discrepancy_loss": float(discrepancy.detach().cpu()),
        "position_loss": float(position.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics


def compute_supervised_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    physics_weight: float,
    discrepancy_weight: float,
    use_uq: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    mean = outputs["regression_mean"]
    if use_uq:
        logvar = outputs["regression_logvar"].clamp(min=-6.0, max=6.0)
        regression = 0.5 * ((targets - mean).pow(2) * torch.exp(-logvar) + logvar).mean()
    else:
        regression = F.mse_loss(mean, targets)
    physics_loss = F.mse_loss(outputs["physics_only_reconstruction"], outputs["reconstruction"].detach()) + second_derivative(outputs["state"]).pow(2).mean()
    discrepancy = outputs["discrepancy"].pow(2).mean()
    total = regression + physics_weight * physics_loss + discrepancy_weight * discrepancy
    metrics = {
        "regression_loss": float(regression.detach().cpu()),
        "physics_loss": float(physics_loss.detach().cpu()),
        "discrepancy_loss": float(discrepancy.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics
