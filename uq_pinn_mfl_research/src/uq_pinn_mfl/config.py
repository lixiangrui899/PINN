from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProjectContext:
    project_root: Path
    config_path: Path
    data_root: Path
    metadata_dir: Path
    reports_dir: Path
    manifests_dir: Path
    outputs_dir: Path
    cache_dir: Path
    checkpoints_dir: Path
    predictions_dir: Path
    config: dict[str, Any]

    @property
    def label_path(self) -> Path:
        return self.metadata_dir / "defect_labels.csv"

    @property
    def label_template_path(self) -> Path:
        return self.metadata_dir / "defect_labels_template.csv"

    @property
    def dataset_index_path(self) -> Path:
        return self.reports_dir / "dataset_index.csv"

    @property
    def pairing_report_path(self) -> Path:
        return self.reports_dir / "pairing_report.csv"

    @property
    def anomaly_report_path(self) -> Path:
        return self.reports_dir / "naming_anomaly_report.csv"

    @property
    def readiness_report_path(self) -> Path:
        return self.reports_dir / "data_readiness_report.md"


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_context(config_path: str | Path) -> ProjectContext:
    config_file = Path(config_path).resolve()
    raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    project_root = _resolve_path(config_file.parent, raw_config.get("project_root", ".."))
    paths = raw_config["paths"]

    context = ProjectContext(
        project_root=project_root,
        config_path=config_file,
        data_root=_resolve_path(project_root, raw_config["data_root"]),
        metadata_dir=_resolve_path(project_root, paths["metadata_dir"]),
        reports_dir=_resolve_path(project_root, paths["reports_dir"]),
        manifests_dir=_resolve_path(project_root, paths["manifests_dir"]),
        outputs_dir=_resolve_path(project_root, paths["outputs_dir"]),
        cache_dir=_resolve_path(project_root, paths["cache_dir"]),
        checkpoints_dir=_resolve_path(project_root, paths["checkpoints_dir"]),
        predictions_dir=_resolve_path(project_root, paths["predictions_dir"]),
        config=raw_config,
    )

    for directory in (
        context.metadata_dir,
        context.reports_dir,
        context.manifests_dir,
        context.outputs_dir,
        context.cache_dir,
        context.checkpoints_dir,
        context.predictions_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return context
