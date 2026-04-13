from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd

from uq_pinn_mfl.config import ProjectContext

REQUIRED_LABEL_COLUMNS = [
    "defect_unit_id",
    "specimen_id",
    "position_id",
    "orientation",
    "run_id",
    "timestamp",
    "length",
    "width",
    "depth",
    "area",
    "defect_severity",
    "label_source",
    "label_reliability",
    "comment",
]


class ProjectMode(StrEnum):
    WEAKLY_SUPERVISED = "weakly_supervised_proxy_mode"
    SUPERVISED = "supervised_inversion_mode"


def generate_label_template(dataset_index: pd.DataFrame, output_path: Path) -> Path:
    template = dataset_index[
        [
            "defect_unit_id",
            "specimen_id",
            "position_id",
            "orientation",
            "normalized_run_id",
            "timestamp",
        ]
    ].copy()
    template.rename(columns={"normalized_run_id": "run_id"}, inplace=True)
    template["length"] = pd.NA
    template["width"] = pd.NA
    template["depth"] = pd.NA
    template["area"] = pd.NA
    template["defect_severity"] = pd.NA
    template["label_source"] = pd.NA
    template["label_reliability"] = pd.NA
    template["comment"] = pd.NA
    template = template[REQUIRED_LABEL_COLUMNS].drop_duplicates().sort_values(
        ["position_id", "orientation", "run_id", "timestamp"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def resolve_project_mode(context: ProjectContext) -> tuple[ProjectMode, list[str]]:
    warnings: list[str] = []
    if not context.label_path.exists():
        warnings.append("metadata/defect_labels.csv is missing. Falling back to weakly supervised / proxy mode.")
        return ProjectMode.WEAKLY_SUPERVISED, warnings

    labels = pd.read_csv(context.label_path)
    missing_columns = [column for column in REQUIRED_LABEL_COLUMNS if column not in labels.columns]
    if missing_columns:
        warnings.append(
            "metadata/defect_labels.csv exists but misses required columns: "
            + ", ".join(missing_columns)
            + ". Falling back to weakly supervised / proxy mode."
        )
        return ProjectMode.WEAKLY_SUPERVISED, warnings

    target_columns = ["length", "width", "depth", "area", "defect_severity"]
    non_empty_targets = any(labels[column].notna().any() for column in target_columns if column in labels.columns)
    if not non_empty_targets:
        warnings.append("Label file exists but target columns are empty. Falling back to weakly supervised / proxy mode.")
        return ProjectMode.WEAKLY_SUPERVISED, warnings

    return ProjectMode.SUPERVISED, warnings
