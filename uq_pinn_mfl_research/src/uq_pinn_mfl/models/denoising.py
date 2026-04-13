from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        x = F.gelu(self.norm1(self.conv1(inputs)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class ResUNet1D(nn.Module):
    def __init__(self, in_channels: int = 8, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ResidualBlock1D(in_channels, base_channels)
        self.enc2 = ResidualBlock1D(base_channels, base_channels * 2)
        self.enc3 = ResidualBlock1D(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bottleneck = ResidualBlock1D(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock1D(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock1D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock1D(base_channels * 2, base_channels)
        self.out = nn.Conv1d(base_channels, in_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([_match_length(d3, e3), e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([_match_length(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([_match_length(d1, e1), e1], dim=1))
        return self.out(d1)


def _match_length(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[-1] == target.shape[-1]:
        return source
    if source.shape[-1] > target.shape[-1]:
        return source[..., : target.shape[-1]]
    return F.pad(source, (0, target.shape[-1] - source.shape[-1]))


def moving_average_denoise(inputs: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    padding = kernel_size // 2
    weight = torch.ones(inputs.shape[1], 1, kernel_size, device=inputs.device) / kernel_size
    return F.conv1d(inputs, weight, padding=padding, groups=inputs.shape[1])


def compute_denoising_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = F.l1_loss(prediction, target)
    grad_pred = prediction[..., 1:] - prediction[..., :-1]
    grad_target = target[..., 1:] - target[..., :-1]
    gradient = F.l1_loss(grad_pred, grad_target)

    peak_pred = F.max_pool1d(prediction, kernel_size=9, stride=1, padding=4)
    peak_target = F.max_pool1d(target, kernel_size=9, stride=1, padding=4)
    valley_pred = -F.max_pool1d(-prediction, kernel_size=9, stride=1, padding=4)
    valley_target = -F.max_pool1d(-target, kernel_size=9, stride=1, padding=4)
    peak_valley = 0.5 * (F.l1_loss(peak_pred, peak_target) + F.l1_loss(valley_pred, valley_target))

    spec_pred = torch.fft.rfft(prediction, dim=-1)
    spec_target = torch.fft.rfft(target, dim=-1)
    spectral = F.l1_loss(torch.abs(spec_pred), torch.abs(spec_target))

    total = (
        loss_weights["reconstruction"] * reconstruction
        + loss_weights["gradient"] * gradient
        + loss_weights["peak_valley"] * peak_valley
        + loss_weights["spectral"] * spectral
    )
    metrics = {
        "reconstruction": float(reconstruction.detach().cpu()),
        "gradient": float(gradient.detach().cpu()),
        "peak_valley": float(peak_valley.detach().cpu()),
        "spectral": float(spectral.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics
