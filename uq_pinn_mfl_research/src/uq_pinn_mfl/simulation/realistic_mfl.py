from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

from uq_pinn_mfl.config import ProjectContext

AIN_COLUMNS = [f"AIN {index}" for index in range(1, 9)]
FILE_RE = re.compile(r"^(?P<run>\d+)[_-](?P<timestamp>\d{14})[_-](?P<tag>[A-Za-z0-9]+)$")


@dataclass(slots=True)
class ReferenceStats:
    axial_mean: np.ndarray
    axial_std: np.ndarray
    radial_mean: np.ndarray
    radial_std: np.ndarray
    channel_bias: np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _parse_layout_entry(csv_path: Path, root: Path) -> dict[str, Any]:
    rel = csv_path.relative_to(root)
    position_name, orientation_name = rel.parts[0], rel.parts[1]
    position_id = int(re.sub(r"\D", "", position_name))
    orientation = "axial" if "轴" in orientation_name or "ax" in orientation_name.lower() else "radial"
    match = FILE_RE.match(csv_path.stem)
    if not match:
        raise ValueError(f"Unsupported filename pattern: {csv_path.name}")
    index_series = pd.read_csv(csv_path, usecols=["Index"], encoding="utf-8-sig")["Index"]
    unique_points = int(index_series.nunique())
    cycles = int(len(index_series) / unique_points)
    return {
        "position_id": position_id,
        "orientation": orientation,
        "run_id": int(match.group("run")),
        "timestamp": match.group("timestamp"),
        "acquisition_tag": match.group("tag"),
        "scan_points": unique_points,
        "cycles": cycles,
    }


def load_reference_layout(data_root: Path) -> pd.DataFrame:
    rows = [_parse_layout_entry(csv_path, data_root) for csv_path in sorted(data_root.rglob("*.csv"))]
    return pd.DataFrame(rows).sort_values(["position_id", "orientation", "run_id"]).reset_index(drop=True)


def estimate_reference_statistics(data_root: Path, sample_rows: int = 12000) -> ReferenceStats:
    stats: dict[str, list[np.ndarray]] = {"axial_mean": [], "axial_std": [], "radial_mean": [], "radial_std": []}
    bias_rows: list[np.ndarray] = []
    for csv_path in sorted(data_root.rglob("*.csv")):
        rel = csv_path.relative_to(data_root)
        orientation = "axial" if "轴" in rel.parts[1] else "radial"
        frame = pd.read_csv(csv_path, nrows=sample_rows, usecols=AIN_COLUMNS, encoding="utf-8-sig")
        values = frame.to_numpy(dtype=np.float32)
        stats[f"{orientation}_mean"].append(values.mean(axis=0))
        stats[f"{orientation}_std"].append(values.std(axis=0) + 1e-6)
        bias_rows.append(values.mean(axis=0))

    axial_mean = np.vstack(stats["axial_mean"]).mean(axis=0)
    axial_std = np.vstack(stats["axial_std"]).mean(axis=0)
    radial_mean = np.vstack(stats["radial_mean"]).mean(axis=0)
    radial_std = np.vstack(stats["radial_std"]).mean(axis=0)
    channel_bias = np.vstack(bias_rows).mean(axis=0)
    return ReferenceStats(
        axial_mean=axial_mean.astype(np.float32),
        axial_std=axial_std.astype(np.float32),
        radial_mean=radial_mean.astype(np.float32),
        radial_std=radial_std.astype(np.float32),
        channel_bias=channel_bias.astype(np.float32),
    )


def _charge_pair_field(x: np.ndarray, center: float, half_length: float, lift_off: float) -> np.ndarray:
    left = x - (center - half_length)
    right = x - (center + half_length)
    left_field = left / np.power(left**2 + lift_off**2, 1.5)
    right_field = right / np.power(right**2 + lift_off**2, 1.5)
    return left_field - right_field


def _gaussian(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / max(sigma, 1e-4)) ** 2)


def _build_core_signal(
    x: np.ndarray,
    position_code: int,
    orientation: str,
    defect: dict[str, float],
    channel_offsets: np.ndarray,
) -> np.ndarray:
    base_center = defect["center"]
    length_norm = 0.06 + 0.18 * (defect["length"] - 6.0) / (28.0 - 6.0)
    width_norm = 0.02 + 0.07 * (defect["width"] - 1.5) / (8.0 - 1.5)
    lift_norm = 0.03 + 0.04 * (defect["lift_off"] - 0.8) / (2.5 - 0.8)
    strength = 0.8 + 0.18 * defect["depth"] + 0.035 * defect["length"] + 0.05 * defect["width"]
    strength *= defect["magnetization_scale"]
    half_length = 0.5 * length_norm

    channels = []
    for sensor_idx, offset in enumerate(channel_offsets):
        local_center = base_center + 0.5 * offset + 0.01 * np.sin(position_code + sensor_idx)
        pair_field = _charge_pair_field(x, local_center, half_length, lift_norm + 0.02 * abs(offset))
        bowl = -0.55 * _gaussian(x, local_center, sigma=width_norm + 0.01 * sensor_idx)
        gradient = np.gradient(pair_field, x)
        if orientation == "axial":
            core = 0.82 * pair_field + 0.18 * bowl
        else:
            core = 0.58 * gradient + 0.25 * pair_field + 0.17 * bowl
        channels.append(core * (strength * (1.0 + 0.05 * sensor_idx)))
    return np.vstack(channels).astype(np.float32)


def _make_cycle(
    scan_points: int,
    orientation: str,
    defect: dict[str, float],
    stats: ReferenceStats,
    rng: np.random.Generator,
    cycle_idx: int,
    position_code: int,
) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, scan_points, dtype=np.float32)
    channel_offsets = np.linspace(-0.12, 0.12, 8, dtype=np.float32)
    defect_with_jitter = dict(defect)
    defect_with_jitter["center"] = defect["center"] + rng.normal(0.0, 0.01)
    defect_with_jitter["magnetization_scale"] = defect["magnetization_scale"] * (1.0 + rng.normal(0.0, 0.025))

    core = _build_core_signal(x, position_code=position_code, orientation=orientation, defect=defect_with_jitter, channel_offsets=channel_offsets)
    baseline_mean = stats.axial_mean if orientation == "axial" else stats.radial_mean
    baseline_std = stats.axial_std if orientation == "axial" else stats.radial_std
    target_amp = baseline_std * (1.8 + 0.4 * defect["depth"])
    core_std = core.std(axis=1, keepdims=True)
    scaled_core = core * (target_amp[:, None] / np.maximum(core_std, 1e-5))

    low_freq = (
        0.25 * np.sin(2.0 * np.pi * x[None,] * (1.0 + 0.1 * cycle_idx) + rng.uniform(0, np.pi))
        + 0.12 * np.cos(np.pi * x[None,] * (2.0 + 0.15 * position_code))
    )
    low_freq = low_freq * baseline_std[:, None]
    common_mode = rng.normal(0.0, 0.15, size=(1, scan_points)).astype(np.float32) * baseline_std.mean()
    channel_noise = rng.normal(0.0, baseline_std[:, None] * 0.22, size=(8, scan_points)).astype(np.float32)
    cycle_bias = rng.normal(0.0, baseline_std * 0.08, size=8).astype(np.float32)

    signal = baseline_mean[:, None] + cycle_bias[:, None] + 0.22 * low_freq + common_mode + scaled_core + channel_noise
    return signal.astype(np.float32)


def _assemble_signal_file(
    scan_points: int,
    cycles: int,
    orientation: str,
    defect: dict[str, float],
    stats: ReferenceStats,
    rng: np.random.Generator,
    position_code: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    cycle_frames = []
    clean_cycles = []
    for cycle_idx in range(cycles):
        cycle_signal = _make_cycle(scan_points, orientation, defect, stats, rng, cycle_idx, position_code)
        clean_cycles.append(cycle_signal)
        cycle_frame = pd.DataFrame(cycle_signal.T, columns=AIN_COLUMNS)
        cycle_frame.insert(0, "Index", np.arange(1, scan_points + 1, dtype=np.int32))
        cycle_frames.append(cycle_frame)
    frame = pd.concat(cycle_frames, ignore_index=True)
    clean = np.concatenate(clean_cycles, axis=1)
    return frame, clean


def _sample_defect_specs(seed: int, num_positions: int, priors: dict[str, float]) -> dict[int, dict[str, float]]:
    rng = _rng(seed)
    specs: dict[int, dict[str, float]] = {}
    for position_id in range(1, num_positions + 1):
        length = rng.uniform(priors["length_min"], priors["length_max"])
        width = rng.uniform(priors["width_min"], priors["width_max"])
        depth = rng.uniform(priors["depth_min"], priors["depth_max"])
        lift_off = rng.uniform(priors["lift_off_min"], priors["lift_off_max"])
        severity = min(5.0, 1.0 + (depth / priors["depth_max"]) * 4.0)
        specs[position_id] = {
            "length": round(float(length), 4),
            "width": round(float(width), 4),
            "depth": round(float(depth), 4),
            "area": round(float(length * width), 4),
            "defect_severity": round(float(severity), 4),
            "lift_off": round(float(lift_off), 4),
            "center": float(rng.uniform(-priors["center_jitter"], priors["center_jitter"])),
            "magnetization_scale": float(rng.uniform(0.92, 1.08)),
        }
    return specs


def _build_benchmark_layout(context: ProjectContext) -> pd.DataFrame:
    cfg = context.config["simulation"]["benchmark"]
    rows: list[dict[str, Any]] = []
    base_time = datetime(2025, 8, 3, 9, 0, 0)
    for position_id in range(1, int(cfg["num_positions"]) + 1):
        for run_id in range(1, int(cfg["runs_per_position"]) + 1):
            timestamp = (base_time + timedelta(minutes=position_id * 17 + run_id * 3)).strftime("%Y%m%d%H%M%S")
            rows.append(
                {
                    "position_id": position_id,
                    "orientation": "axial",
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "acquisition_tag": cfg["acquisition_tag"],
                    "scan_points": int(cfg["scan_points"]),
                    "cycles": int(cfg["cycles_min"]) + ((position_id + run_id) % (int(cfg["cycles_max"]) - int(cfg["cycles_min"]) + 1)),
                }
            )
            if bool(cfg["include_radial"]):
                rows.append(
                    {
                        "position_id": position_id,
                        "orientation": "radial",
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "acquisition_tag": cfg["acquisition_tag"],
                        "scan_points": int(cfg["scan_points"]),
                        "cycles": int(cfg["cycles_min"]) + ((position_id + run_id + 1) % (int(cfg["cycles_max"]) - int(cfg["cycles_min"]) + 1)),
                    }
                )
    return pd.DataFrame(rows)


def _write_generated_config(context: ProjectContext, dataset_name: str, synthetic_root: Path, metadata_dir: Path) -> Path:
    base_config = context.config.copy()
    generated = dict(base_config)
    generated["project_root"] = "../.."
    generated["data_root"] = str(synthetic_root)
    generated["paths"] = dict(base_config["paths"])
    generated["paths"]["metadata_dir"] = str(metadata_dir.relative_to(context.project_root)).replace("\\", "/")
    generated["paths"]["reports_dir"] = f"reports/{dataset_name}"
    generated["paths"]["manifests_dir"] = f"manifests/{dataset_name}"
    generated["paths"]["outputs_dir"] = f"outputs/{dataset_name}_pipeline"
    generated["paths"]["cache_dir"] = f"outputs/{dataset_name}_pipeline/cache"
    generated["paths"]["checkpoints_dir"] = f"outputs/{dataset_name}_pipeline/checkpoints"
    generated["paths"]["predictions_dir"] = f"outputs/{dataset_name}_pipeline/predictions"
    generated_path = context.project_root / "configs" / "generated" / f"{dataset_name}.yaml"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(yaml.safe_dump(generated, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return generated_path


def generate_synthetic_dataset(context: ProjectContext, mode: str | None = None, dataset_name: str | None = None) -> dict[str, str]:
    sim_cfg = context.config["simulation"]
    dataset_name = dataset_name or sim_cfg["dataset_name"]
    mode = mode or sim_cfg["mode"]
    reference_layout = load_reference_layout(context.data_root)
    stats = estimate_reference_statistics(context.data_root, sample_rows=int(sim_cfg["reference_sample_rows"]))

    if mode == "match_real_layout":
        layout = reference_layout.copy()
        num_positions = int(layout["position_id"].max())
    else:
        layout = _build_benchmark_layout(context)
        num_positions = int(context.config["simulation"]["benchmark"]["num_positions"])

    output_base = context.outputs_dir / "synthetic_datasets" / dataset_name
    synthetic_root = output_base / sim_cfg["default_defect_type"]
    metadata_dir = output_base / "metadata"
    artifacts_dir = output_base / "artifacts"
    clean_dir = artifacts_dir / "clean_signals"
    synthetic_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    defect_specs = _sample_defect_specs(int(sim_cfg["seed"]), num_positions=num_positions, priors=sim_cfg["defect_priors"])
    rng = _rng(int(sim_cfg["seed"]))
    label_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for _, row in layout.iterrows():
        position_id = int(row["position_id"])
        orientation_folder = "轴向" if row["orientation"] == "axial" else "径向"
        position_folder = synthetic_root / f"位置{position_id}"
        file_dir = position_folder / orientation_folder
        file_dir.mkdir(parents=True, exist_ok=True)

        defect = dict(defect_specs[position_id])
        frame, clean = _assemble_signal_file(
            scan_points=int(row["scan_points"]),
            cycles=int(row["cycles"]),
            orientation=str(row["orientation"]),
            defect=defect,
            stats=stats,
            rng=rng,
            position_code=position_id,
        )
        file_name = f"{int(row['run_id']):02d}_{row['timestamp']}_{row['acquisition_tag']}.csv"
        csv_path = file_dir / file_name
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")

        clean_path = clean_dir / f"position_{position_id:03d}_{row['orientation']}_run_{int(row['run_id']):02d}.npz"
        np.savez_compressed(clean_path, clean_signal=clean)

        defect_unit_id = f"{sim_cfg['default_defect_type']}__position_{position_id}"
        label_rows.append(
            {
                "defect_unit_id": defect_unit_id,
                "specimen_id": sim_cfg["default_defect_type"],
                "position_id": position_id,
                "orientation": row["orientation"],
                "run_id": int(row["run_id"]),
                "timestamp": row["timestamp"],
                "length": defect["length"],
                "width": defect["width"],
                "depth": defect["depth"],
                "area": defect["area"],
                "defect_severity": defect["defect_severity"],
                "label_source": "synthetic_charge_pair_prior",
                "label_reliability": 1.0,
                "comment": f"preset={sim_cfg['preset']}; mode={mode}",
            }
        )
        manifest_rows.append(
            {
                "path": str(csv_path),
                "position_id": position_id,
                "orientation": row["orientation"],
                "run_id": int(row["run_id"]),
                "timestamp": row["timestamp"],
                "acquisition_tag": row["acquisition_tag"],
                "scan_points": int(row["scan_points"]),
                "cycles": int(row["cycles"]),
                "clean_signal_path": str(clean_path),
                "length": defect["length"],
                "width": defect["width"],
                "depth": defect["depth"],
            }
        )

    labels = pd.DataFrame(label_rows).sort_values(["position_id", "orientation", "run_id"])
    labels_path = metadata_dir / "defect_labels.csv"
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig")

    manifest = pd.DataFrame(manifest_rows).sort_values(["position_id", "orientation", "run_id"])
    manifest_path = metadata_dir / "synthetic_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    summary_path = metadata_dir / "simulation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Synthetic MFL Dataset Summary",
                "",
                f"- Mode: `{mode}`",
                f"- Dataset root: `{synthetic_root}`",
                f"- File count: **{len(manifest)}**",
                f"- Position count: **{manifest['position_id'].nunique()}**",
                f"- Orientations: **{', '.join(sorted(manifest['orientation'].unique()))}**",
                f"- Acquisition tags: **{', '.join(sorted(manifest['acquisition_tag'].unique()))}**",
                "- Labels are stored in `defect_labels.csv` and can directly unlock supervised inversion mode via the generated config.",
            ]
        ),
        encoding="utf-8",
    )

    generated_config = _write_generated_config(context, dataset_name, synthetic_root, metadata_dir)
    return {
        "synthetic_data_root": str(synthetic_root),
        "labels_path": str(labels_path),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "generated_config": str(generated_config),
    }
