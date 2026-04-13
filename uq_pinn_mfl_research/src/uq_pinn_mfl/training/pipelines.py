from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uq_pinn_mfl.config import ProjectContext
from uq_pinn_mfl.data.audit import run_audit
from uq_pinn_mfl.data.labels import ProjectMode, generate_label_template, resolve_project_mode
from uq_pinn_mfl.data.preprocess import LabeledSignalDataset, PairedWindowDataset, SignalWindowDataset, preprocess_dataset
from uq_pinn_mfl.data.splits import build_lopo_schedule, write_split_manifests
from uq_pinn_mfl.evaluation.uq import calibration_curve, gaussian_nll, mpiw, picp, prediction_interval
from uq_pinn_mfl.models.denoising import ResUNet1D, compute_denoising_loss, moving_average_denoise
from uq_pinn_mfl.models.surrogate import (
    BlackBoxRegressor,
    BlackBoxProxyNet,
    PhysicsGuidedSurrogate,
    compute_proxy_loss,
    compute_supervised_loss,
)
from uq_pinn_mfl.training.common import choose_train_val_split, resolve_device, save_json, set_seed


@dataclass(slots=True)
class TargetNormalizer:
    target_names: list[str]
    mode: str
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, target_names: list[str], mode: str) -> "TargetNormalizer":
        if mode == "none":
            mean = np.zeros(len(target_names), dtype=np.float32)
            std = np.ones(len(target_names), dtype=np.float32)
            return cls(target_names=target_names, mode=mode, mean=mean, std=std)
        if mode != "zscore":
            raise ValueError(f"Unsupported target normalization mode: {mode}")

        values = frame[target_names].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(target_names=target_names, mode=mode, mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform_tensor(self, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return targets
        mean = torch.as_tensor(self.mean, device=targets.device, dtype=targets.dtype)
        std = torch.as_tensor(self.std, device=targets.device, dtype=targets.dtype)
        return (targets - mean) / std

    def inverse_mean_array(self, mean: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return mean
        return mean * self.std[None, :] + self.mean[None, :]

    def inverse_logvar_array(self, logvar: np.ndarray | None) -> np.ndarray | None:
        if logvar is None or self.mode == "none":
            return logvar
        return logvar + 2.0 * np.log(self.std[None, :])

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mean": {name: float(value) for name, value in zip(self.target_names, self.mean)},
            "std": {name: float(value) for name, value in zip(self.target_names, self.std)},
        }


def run_stage_a(context: ProjectContext) -> dict[str, str]:
    outputs = run_audit(context)
    dataset_index = pd.read_csv(context.dataset_index_path)
    pairing_report = pd.read_csv(context.pairing_report_path) if context.pairing_report_path.exists() else pd.DataFrame()
    generate_label_template(dataset_index, context.label_template_path)
    split_paths = write_split_manifests(context, dataset_index, pairing_report)
    mode, warnings = resolve_project_mode(context)
    mode_payload = {
        "mode": mode,
        "warnings": warnings,
        "label_template": str(context.label_template_path),
        "reports": {name: str(path) for name, path in outputs.items()},
        "manifests": split_paths,
    }
    save_json(context.reports_dir / "project_mode.json", mode_payload)
    return {key: str(value) for key, value in outputs.items()} | {"label_template": str(context.label_template_path)}


def _average_metrics(metric_history: list[dict[str, float]]) -> dict[str, float]:
    if not metric_history:
        return {}
    aggregate: dict[str, float] = defaultdict(float)
    for metrics in metric_history:
        for key, value in metrics.items():
            aggregate[key] += float(value)
    return {key: value / len(metric_history) for key, value in aggregate.items()}


def _run_denoising_epoch(
    loader: DataLoader,
    model: ResUNet1D | None,
    device: torch.device,
    loss_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    max_steps: int | None,
) -> dict[str, float]:
    history: list[dict[str, float]] = []
    for step, batch in enumerate(loader):
        inputs = batch["inputs"].float().to(device)
        targets = batch["targets"].float().to(device)
        if model is None:
            predictions = moving_average_denoise(inputs)
            _, metrics = compute_denoising_loss(predictions, targets, loss_weights)
        else:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss, metrics = compute_denoising_loss(predictions, targets, loss_weights)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        history.append(metrics)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return _average_metrics(history)


def run_stage_b(context: ProjectContext, model_name: str = "resunet1d", rebuild: bool = False) -> dict[str, Any]:
    preprocess_dataset(context, rebuild=rebuild)
    window_frame = pd.read_csv(context.reports_dir / "window_index.csv")
    cfg = context.config
    split_cfg = cfg["splits"]
    train_frame, val_frame, test_frame = choose_train_val_split(
        window_frame,
        holdout_position=int(split_cfg["holdout_position"]),
        validation_position=int(split_cfg["validation_position"]),
    )

    train_loader = DataLoader(SignalWindowDataset(train_frame), batch_size=int(cfg["denoising"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(SignalWindowDataset(val_frame), batch_size=int(cfg["denoising"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(SignalWindowDataset(test_frame), batch_size=int(cfg["denoising"]["batch_size"]), shuffle=False)

    set_seed(int(cfg["seed"]))
    device = resolve_device()
    loss_weights = cfg["denoising"]["loss_weights"]
    max_steps = int(cfg["denoising"]["max_steps"])

    if model_name == "moving_average":
        model = None
        optimizer = None
        checkpoint_path = None
    else:
        model = ResUNet1D().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["denoising"]["lr"]))
        checkpoint_path = context.checkpoints_dir / f"stage_b_{model_name}.pt"
        for _ in range(int(cfg["denoising"]["epochs"])):
            _run_denoising_epoch(train_loader, model, device, loss_weights, optimizer, max_steps)
        torch.save(model.state_dict(), checkpoint_path)

    metrics = {
        "stage": "B",
        "model": model_name,
        "train": _run_denoising_epoch(train_loader, model, device, loss_weights, None, max_steps),
        "val": _run_denoising_epoch(val_loader, model, device, loss_weights, None, max_steps),
        "test": _run_denoising_epoch(test_loader, model, device, loss_weights, None, max_steps),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }
    save_json(context.reports_dir / "stage_b_metrics.json", metrics)
    return metrics


def _proxy_loss_for_variant(
    variant: str,
    outputs: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    position_targets: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if variant == "blackbox":
        reconstruction = F.mse_loss(outputs["reconstruction"], inputs)
        position = F.cross_entropy(outputs["position_logits"], position_targets)
        total = reconstruction + float(cfg["position_weight"]) * position
        return total, {
            "data_loss": float(reconstruction.detach().cpu()),
            "position_loss": float(position.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
    return compute_proxy_loss(
        outputs,
        inputs,
        position_targets,
        physics_weight=float(cfg["physics_weight"]),
        discrepancy_weight=float(cfg["discrepancy_weight"]),
        position_weight=float(cfg["position_weight"]),
    )


def _run_proxy_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    variant: str,
    max_steps: int,
) -> dict[str, float]:
    history: list[dict[str, float]] = []
    for step, batch in enumerate(loader):
        axial = batch["axial"].float().to(device)
        radial = batch["radial"].float().to(device)
        positions = batch["position_id"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        out_axial = model(axial)
        out_radial = model(radial)
        loss_axial, metrics_axial = _proxy_loss_for_variant(variant, out_axial, axial, positions, cfg)
        loss_radial, metrics_radial = _proxy_loss_for_variant(variant, out_radial, radial, positions, cfg)
        consistency = 1.0 - F.cosine_similarity(out_axial["embedding"], out_radial["embedding"]).mean()
        total = loss_axial + loss_radial + float(cfg["consistency_weight"]) * consistency
        if optimizer is not None:
            total.backward()
            optimizer.step()

        merged = {
            key: 0.5 * (metrics_axial.get(key, 0.0) + metrics_radial.get(key, 0.0))
            for key in set(metrics_axial) | set(metrics_radial)
        }
        merged["consistency_loss"] = float(consistency.detach().cpu())
        merged["total"] = float(total.detach().cpu())
        history.append(merged)
        if step + 1 >= max_steps:
            break
    return _average_metrics(history)


def run_stage_c(context: ProjectContext, variant: str = "uq_pinn", rebuild: bool = False) -> dict[str, Any]:
    mode, warnings = resolve_project_mode(context)
    preprocess_dataset(context, rebuild=rebuild)
    paired_frame = pd.read_csv(context.reports_dir / "paired_window_index.csv")
    if paired_frame.empty:
        raise ValueError("No paired axial/radial windows were found. Check pairing_report.csv first.")

    cfg = context.config
    split_cfg = cfg["splits"]
    train_frame, val_frame, test_frame = choose_train_val_split(
        paired_frame,
        holdout_position=int(split_cfg["holdout_position"]),
        validation_position=int(split_cfg["validation_position"]),
        group_column="pair_key",
    )
    train_loader = DataLoader(PairedWindowDataset(train_frame), batch_size=int(cfg["proxy"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(PairedWindowDataset(val_frame), batch_size=int(cfg["proxy"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(PairedWindowDataset(test_frame), batch_size=int(cfg["proxy"]["batch_size"]), shuffle=False)

    set_seed(int(cfg["seed"]))
    device = resolve_device()
    if variant == "blackbox":
        model: torch.nn.Module = BlackBoxProxyNet().to(device)
    elif variant == "deterministic_pinn":
        model = PhysicsGuidedSurrogate(use_uq=False, use_discrepancy=False).to(device)
    else:
        model = PhysicsGuidedSurrogate(use_uq=True, use_discrepancy=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["proxy"]["lr"]))

    for _ in range(int(cfg["proxy"]["epochs"])):
        _run_proxy_epoch(
            train_loader,
            model,
            device,
            cfg["proxy"],
            optimizer,
            variant,
            max_steps=int(cfg["proxy"]["max_steps"]),
        )

    checkpoint_path = context.checkpoints_dir / f"stage_c_{variant}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    metrics = {
        "stage": "C",
        "variant": variant,
        "mode": mode,
        "warnings": warnings,
        "train": _run_proxy_epoch(train_loader, model, device, cfg["proxy"], None, variant, int(cfg["proxy"]["max_steps"])),
        "val": _run_proxy_epoch(val_loader, model, device, cfg["proxy"], None, variant, int(cfg["proxy"]["max_steps"])),
        "test": _run_proxy_epoch(test_loader, model, device, cfg["proxy"], None, variant, int(cfg["proxy"]["max_steps"])),
        "checkpoint": str(checkpoint_path),
    }
    save_json(context.reports_dir / "stage_c_metrics.json", metrics)
    return metrics


def _load_supervised_frame(context: ProjectContext) -> pd.DataFrame:
    preprocessed = pd.read_csv(context.reports_dir / "preprocessed_index.csv")
    labels = pd.read_csv(context.label_path)
    labels["run_id"] = labels["run_id"].astype(str)
    labels["position_id"] = labels["position_id"].astype(int)
    labels["timestamp"] = labels["timestamp"].astype(str)
    preprocessed["merge_run_id"] = preprocessed["normalized_run_id"].astype(str)
    preprocessed["timestamp"] = preprocessed["timestamp"].astype(str)
    merged = preprocessed.merge(
        labels,
        left_on=["defect_unit_id", "position_id", "orientation", "merge_run_id", "timestamp"],
        right_on=["defect_unit_id", "position_id", "orientation", "run_id", "timestamp"],
        how="inner",
    )
    return merged


def _evaluate_supervised_predictions(
    targets: np.ndarray,
    mean: np.ndarray,
    logvar: np.ndarray | None,
    uq_cfg: dict[str, Any],
    target_names: list[str],
) -> dict[str, Any]:
    if targets.size == 0 or mean.size == 0:
        return {"sample_count": 0, "mae": None, "rmse": None, "per_target": {}}

    metrics = {
        "sample_count": int(targets.shape[0]),
        "mae": float(np.mean(np.abs(targets - mean))),
        "rmse": float(np.sqrt(np.mean((targets - mean) ** 2))),
        "per_target": {},
    }
    if logvar is not None:
        lower, upper = prediction_interval(mean, logvar, z_score=float(uq_cfg["confidence_z"]))
        metrics["nll"] = gaussian_nll(targets, mean, logvar)
        metrics["picp"] = picp(targets, lower, upper)
        metrics["mpiw"] = mpiw(lower, upper)
        metrics["calibration_curve"] = calibration_curve(
            targets,
            mean,
            logvar,
            bins=int(uq_cfg["calibration_bins"]),
        )
    for index, target_name in enumerate(target_names):
        target_values = targets[:, index]
        mean_values = mean[:, index]
        target_metrics: dict[str, Any] = {
            "mae": float(np.mean(np.abs(target_values - mean_values))),
            "rmse": float(np.sqrt(np.mean((target_values - mean_values) ** 2))),
        }
        if logvar is not None:
            target_logvar = logvar[:, index]
            lower, upper = prediction_interval(mean_values, target_logvar, z_score=float(uq_cfg["confidence_z"]))
            target_metrics["nll"] = gaussian_nll(target_values, mean_values, target_logvar)
            target_metrics["picp"] = picp(target_values, lower, upper)
            target_metrics["mpiw"] = mpiw(lower, upper)
            target_metrics["calibration_curve"] = calibration_curve(
                target_values,
                mean_values,
                target_logvar,
                bins=int(uq_cfg["calibration_bins"]),
            )
        metrics["per_target"][target_name] = target_metrics
    return metrics


def _supervised_loss_for_variant(
    variant: str,
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    physics_weight: float,
    discrepancy_weight: float,
    use_uq: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if variant == "blackbox":
        mean = outputs["regression_mean"]
        regression = F.mse_loss(mean, targets)
        return regression, {"regression_loss": float(regression.detach().cpu()), "total": float(regression.detach().cpu())}
    return compute_supervised_loss(
        outputs,
        targets,
        physics_weight=physics_weight,
        discrepancy_weight=discrepancy_weight,
        use_uq=use_uq,
    )


def _build_supervised_loaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_columns: list[str],
    dataset_cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(
            LabeledSignalDataset(train_frame, target_columns, int(dataset_cfg["downsample_length"])),
            batch_size=int(dataset_cfg["batch_size"]),
            shuffle=True,
        ),
        DataLoader(
            LabeledSignalDataset(val_frame, target_columns, int(dataset_cfg["downsample_length"])),
            batch_size=int(dataset_cfg["batch_size"]),
            shuffle=False,
        ),
        DataLoader(
            LabeledSignalDataset(test_frame, target_columns, int(dataset_cfg["downsample_length"])),
            batch_size=int(dataset_cfg["batch_size"]),
            shuffle=False,
        ),
    )


def _create_stage_d_model(
    variant: str,
    target_dim: int,
    context: ProjectContext,
    device: torch.device,
) -> tuple[torch.nn.Module, bool, float, float]:
    use_uq = variant == "uq_pinn"
    if variant == "blackbox":
        model = BlackBoxRegressor(target_dim=target_dim, use_uq=False).to(device)
        return model, False, 0.0, 0.0
    if variant == "deterministic_pinn":
        model = PhysicsGuidedSurrogate(target_dim=target_dim, use_uq=False, use_discrepancy=False).to(device)
        return model, False, float(context.config["proxy"]["physics_weight"]), 0.0
    model = PhysicsGuidedSurrogate(target_dim=target_dim, use_uq=True, use_discrepancy=True).to(device)
    return model, use_uq, float(context.config["proxy"]["physics_weight"]), float(context.config["proxy"]["discrepancy_weight"])


def _collect_stage_d_predictions(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    target_columns: list[str],
    normalizer: TargetNormalizer,
    use_uq: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    means: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].float().to(device)
            outputs = model(inputs)
            means.append(outputs["regression_mean"].detach().cpu().numpy())
            targets.append(batch["targets"].numpy())
            if use_uq:
                logvars.append(outputs["regression_logvar"].detach().cpu().numpy())
    mean_array = np.concatenate(means, axis=0) if means else np.empty((0, len(target_columns)))
    target_array = np.concatenate(targets, axis=0) if targets else np.empty((0, len(target_columns)))
    logvar_array = np.concatenate(logvars, axis=0) if use_uq and logvars else None
    return (
        target_array,
        normalizer.inverse_mean_array(mean_array),
        normalizer.inverse_logvar_array(logvar_array),
    )


def _write_failure_cases(
    output_path: Path,
    test_frame: pd.DataFrame,
    targets: np.ndarray,
    mean: np.ndarray,
    target_columns: list[str],
) -> None:
    if len(targets) == 0:
        return
    failures = pd.DataFrame({"raw_file_id": test_frame["raw_file_id"].reset_index(drop=True)})
    for index, target_name in enumerate(target_columns):
        failures[f"truth_{target_name}"] = targets[:, index]
        failures[f"pred_{target_name}"] = mean[:, index]
        failures[f"abs_error_{target_name}"] = np.abs(targets[:, index] - mean[:, index])
    failures["mean_absolute_error"] = failures[[f"abs_error_{target}" for target in target_columns]].mean(axis=1)
    failures.sort_values("mean_absolute_error", ascending=False).to_csv(output_path, index=False, encoding="utf-8-sig")


def _summarize_scalar_list(values: list[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summarize_calibration_curves(curves: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    if not curves:
        return []
    summary = []
    for index in range(len(curves[0])):
        nominal = curves[0][index]["nominal"]
        empirical_values = [curve[index]["empirical"] for curve in curves]
        empirical_summary = _summarize_scalar_list(empirical_values)
        summary.append(
            {
                "nominal": float(nominal),
                "empirical_mean": empirical_summary["mean"],
                "empirical_std": empirical_summary["std"],
            }
        )
    return summary


def _summarize_subset_metrics(subset_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not subset_metrics:
        return {}
    summary: dict[str, Any] = {}
    scalar_keys = [key for key, value in subset_metrics[0].items() if isinstance(value, (int, float))]
    for key in scalar_keys:
        summary[key] = _summarize_scalar_list([metrics[key] for metrics in subset_metrics if metrics.get(key) is not None])
    if "calibration_curve" in subset_metrics[0]:
        summary["calibration_curve"] = _summarize_calibration_curves(
            [metrics["calibration_curve"] for metrics in subset_metrics if metrics.get("calibration_curve")]
        )
    per_target_summary: dict[str, Any] = {}
    target_names = subset_metrics[0].get("per_target", {}).keys()
    for target_name in target_names:
        target_metrics = [metrics["per_target"][target_name] for metrics in subset_metrics]
        target_summary: dict[str, Any] = {}
        target_scalar_keys = [key for key, value in target_metrics[0].items() if isinstance(value, (int, float))]
        for key in target_scalar_keys:
            target_summary[key] = _summarize_scalar_list([metrics[key] for metrics in target_metrics if metrics.get(key) is not None])
        if "calibration_curve" in target_metrics[0]:
            target_summary["calibration_curve"] = _summarize_calibration_curves(
                [metrics["calibration_curve"] for metrics in target_metrics if metrics.get("calibration_curve")]
            )
        per_target_summary[target_name] = target_summary
    if per_target_summary:
        summary["per_target"] = per_target_summary
    return summary


def _run_stage_d_fold(
    context: ProjectContext,
    supervised_frame: pd.DataFrame,
    target_columns: list[str],
    variant: str,
    holdout_position: int,
    validation_position: int,
    fold_name: str,
) -> dict[str, Any]:
    train_frame, val_frame, test_frame = choose_train_val_split(
        supervised_frame,
        holdout_position=holdout_position,
        validation_position=validation_position,
    )
    if train_frame.empty:
        raise ValueError(f"Training frame is empty for fold '{fold_name}'.")

    dataset_cfg = context.config["supervised"]
    normalizer = TargetNormalizer.fit(train_frame, target_columns, mode=str(dataset_cfg.get("target_normalization", "zscore")))
    train_loader, val_loader, test_loader = _build_supervised_loaders(train_frame, val_frame, test_frame, target_columns, dataset_cfg)

    set_seed(int(context.config["seed"]) + int(holdout_position))
    device = resolve_device()
    model, use_uq, physics_weight, discrepancy_weight = _create_stage_d_model(
        variant,
        target_dim=len(target_columns),
        context=context,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(dataset_cfg["lr"]))
    max_steps = int(dataset_cfg["max_steps"])
    model.train()
    for _ in range(int(dataset_cfg["epochs"])):
        for step, batch in enumerate(train_loader):
            inputs = batch["inputs"].float().to(device)
            targets = normalizer.transform_tensor(batch["targets"].float().to(device))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss, _ = _supervised_loss_for_variant(
                variant,
                outputs,
                targets,
                physics_weight,
                discrepancy_weight,
                use_uq,
            )
            loss.backward()
            optimizer.step()
            if step + 1 >= max_steps:
                break

    checkpoint_path = context.checkpoints_dir / f"stage_d_{variant}__{fold_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    train_targets, train_mean, train_logvar = _collect_stage_d_predictions(train_loader, model, device, target_columns, normalizer, use_uq)
    val_targets, val_mean, val_logvar = _collect_stage_d_predictions(val_loader, model, device, target_columns, normalizer, use_uq)
    test_targets, test_mean, test_logvar = _collect_stage_d_predictions(test_loader, model, device, target_columns, normalizer, use_uq)

    failure_path = context.reports_dir / f"failure_cases_{variant}__{fold_name}.csv"
    _write_failure_cases(failure_path, test_frame, test_targets, test_mean, target_columns)

    return {
        "fold": fold_name,
        "split": {
            "holdout_position": int(holdout_position),
            "validation_position": int(validation_position),
            "train_positions": sorted(train_frame["position_id"].unique().tolist()),
            "val_positions": sorted(val_frame["position_id"].unique().tolist()),
            "test_positions": sorted(test_frame["position_id"].unique().tolist()),
        },
        "target_normalization": normalizer.to_payload(),
        "train": _evaluate_supervised_predictions(train_targets, train_mean, train_logvar, context.config["uq"], target_columns),
        "val": _evaluate_supervised_predictions(val_targets, val_mean, val_logvar, context.config["uq"], target_columns),
        "test": _evaluate_supervised_predictions(test_targets, test_mean, test_logvar, context.config["uq"], target_columns),
        "checkpoint": str(checkpoint_path),
        "failure_cases": str(failure_path),
    }


def run_stage_d(context: ProjectContext, variant: str = "uq_pinn", rebuild: bool = False, lopo: bool = False) -> dict[str, Any]:
    mode, warnings = resolve_project_mode(context)
    if mode != ProjectMode.SUPERVISED:
        raise ValueError("Supervised inversion mode is blocked until metadata/defect_labels.csv is present and populated.")

    preprocess_dataset(context, rebuild=rebuild)
    supervised_frame = _load_supervised_frame(context)
    target_columns = context.config["supervised"]["target_columns"]
    supervised_frame = supervised_frame.dropna(subset=target_columns).copy()
    if supervised_frame.empty:
        raise ValueError("No labeled samples remain after merging defect_labels.csv with dataset_index.csv.")

    split_cfg = context.config["splits"]
    if not lopo:
        fold_metrics = _run_stage_d_fold(
            context=context,
            supervised_frame=supervised_frame,
            target_columns=target_columns,
            variant=variant,
            holdout_position=int(split_cfg["holdout_position"]),
            validation_position=int(split_cfg["validation_position"]),
            fold_name=f"holdout_position_{int(split_cfg['holdout_position'])}",
        )
        metrics = {
            "stage": "D",
            "variant": variant,
            "mode": mode,
            "warnings": warnings,
            "targets": target_columns,
            "target_normalization": fold_metrics["target_normalization"],
            "split": fold_metrics["split"],
            "train": fold_metrics["train"],
            "val": fold_metrics["val"],
            "test": fold_metrics["test"],
            "checkpoint": fold_metrics["checkpoint"],
            "failure_cases": fold_metrics["failure_cases"],
        }
        save_json(context.reports_dir / "stage_d_metrics.json", metrics)
        return metrics

    positions = supervised_frame["position_id"].dropna().astype(int).unique().tolist()
    lopo_schedule = build_lopo_schedule(positions, preferred_validation_position=int(split_cfg["validation_position"]))
    folds = []
    for schedule in lopo_schedule:
        holdout_position = int(schedule["holdout_position"])
        validation_position = int(schedule["validation_position"])
        folds.append(
            _run_stage_d_fold(
                context=context,
                supervised_frame=supervised_frame,
                target_columns=target_columns,
                variant=variant,
                holdout_position=holdout_position,
                validation_position=validation_position,
                fold_name=f"holdout_position_{holdout_position}",
            )
        )

    metrics = {
        "stage": "D",
        "variant": variant,
        "evaluation": "leave_one_position_out",
        "mode": mode,
        "warnings": warnings,
        "targets": target_columns,
        "folds": folds,
        "summary": {
            "train": _summarize_subset_metrics([fold["train"] for fold in folds]),
            "val": _summarize_subset_metrics([fold["val"] for fold in folds]),
            "test": _summarize_subset_metrics([fold["test"] for fold in folds]),
        },
    }
    save_json(context.reports_dir / f"stage_d_lopo_{variant}.json", metrics)
    return metrics
