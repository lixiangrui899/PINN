from __future__ import annotations

import csv
import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from uq_pinn_mfl.config import ProjectContext

FILE_RE = re.compile(
    r"^(?P<run>\d+)"
    r"(?:[_-](?P<timestamp>\d{14}))?"
    r"(?:[_-](?P<tag>[A-Za-z0-9]+))?$"
)
POSITION_RE = re.compile(r"位置\s*(?P<position>\d+)")
ORIENTATION_ALIASES = {
    "轴向": "axial",
    "径向": "radial",
    "axial": "axial",
    "radial": "radial",
    "ax": "axial",
    "rd": "radial",
}


def _path_token(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]


def _to_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def normalize_orientation(raw_orientation: str) -> tuple[str | None, list[dict[str, Any]]]:
    cleaned = raw_orientation.strip().lower()
    if cleaned in ORIENTATION_ALIASES:
        return ORIENTATION_ALIASES[cleaned], []
    return None, [
        {
            "anomaly_type": "unknown_orientation",
            "severity": "high",
            "message": f"Unable to normalize orientation folder '{raw_orientation}'.",
        }
    ]


def parse_position_id(raw_position: str) -> tuple[int | None, list[dict[str, Any]]]:
    match = POSITION_RE.search(raw_position)
    if match:
        return int(match.group("position")), []
    return None, [
        {
            "anomaly_type": "unknown_position",
            "severity": "high",
            "message": f"Unable to parse position folder '{raw_position}'.",
        }
    ]


def parse_filename(stem: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    match = FILE_RE.match(stem)
    anomalies: list[dict[str, Any]] = []
    parsed: dict[str, Any] = {
        "run_id_raw": None,
        "normalized_run_id": None,
        "timestamp": None,
        "timestamp_iso": None,
        "acquisition_tag": None,
    }
    if not match:
        anomalies.append(
            {
                "anomaly_type": "filename_parse_failed",
                "severity": "high",
                "message": f"Filename stem '{stem}' does not match expected pattern.",
            }
        )
        return parsed, anomalies

    run_id_raw = match.group("run")
    parsed["run_id_raw"] = run_id_raw
    parsed["normalized_run_id"] = int(run_id_raw) if run_id_raw is not None else None
    parsed["timestamp"] = match.group("timestamp")
    parsed["acquisition_tag"] = match.group("tag")
    if parsed["timestamp"]:
        try:
            parsed["timestamp_iso"] = datetime.strptime(parsed["timestamp"], "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            anomalies.append(
                {
                    "anomaly_type": "invalid_timestamp",
                    "severity": "medium",
                    "message": f"Timestamp '{parsed['timestamp']}' is not a valid YYYYmmddHHMMSS value.",
                }
            )
    else:
        anomalies.append(
            {
                "anomaly_type": "missing_timestamp",
                "severity": "medium",
                "message": f"Filename stem '{stem}' is missing the timestamp segment.",
            }
        )

    if parsed["acquisition_tag"] is None:
        anomalies.append(
            {
                "anomaly_type": "missing_acquisition_tag",
                "severity": "medium",
                "message": f"Filename stem '{stem}' is missing an acquisition tag.",
            }
        )
    return parsed, anomalies


def inspect_csv(path: Path, expected_header: list[str], scan_row_count: bool) -> tuple[list[str], int | None]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        row_count = sum(1 for _ in reader) if scan_row_count else None
    return header, row_count


def scan_dataset(context: ProjectContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not context.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {context.data_root}")

    cfg = context.config["audit"]
    expected_header = cfg["expected_header"]
    scan_row_count = bool(cfg.get("scan_row_count", True))
    records: list[dict[str, Any]] = []
    anomaly_records: list[dict[str, Any]] = []

    for csv_path in sorted(context.data_root.rglob("*.csv")):
        relative = csv_path.relative_to(context.data_root)
        relative_parts = relative.parts
        file_anomalies: list[dict[str, Any]] = []
        defect_type = context.data_root.name
        position_id: int | None = None
        orientation: str | None = None

        if len(relative_parts) >= 3:
            position_id, pos_anomalies = parse_position_id(relative_parts[0])
            orientation, ori_anomalies = normalize_orientation(relative_parts[1])
            file_anomalies.extend(pos_anomalies)
            file_anomalies.extend(ori_anomalies)
        else:
            file_anomalies.append(
                {
                    "anomaly_type": "unexpected_path_depth",
                    "severity": "high",
                    "message": f"Path '{relative.as_posix()}' does not match position/orientation/file structure.",
                }
            )

        parsed_name, name_anomalies = parse_filename(csv_path.stem)
        file_anomalies.extend(name_anomalies)
        header, row_count = inspect_csv(csv_path, expected_header, scan_row_count)
        if header != expected_header:
            file_anomalies.append(
                {
                    "anomaly_type": "unexpected_header",
                    "severity": "high",
                    "message": f"Header mismatch. Expected {expected_header}, got {header}.",
                }
            )
        if row_count == 0:
            file_anomalies.append(
                {
                    "anomaly_type": "empty_csv",
                    "severity": "high",
                    "message": f"CSV '{relative.as_posix()}' has zero data rows.",
                }
            )

        relative_path = relative.as_posix()
        raw_file_id = _path_token(relative_path)
        record = {
            "raw_file_id": raw_file_id,
            "defect_type": defect_type,
            "specimen_id": defect_type,
            "defect_unit_id": f"{defect_type}__position_{position_id}" if position_id is not None else None,
            "position_id": position_id,
            "orientation": orientation,
            "filename": csv_path.name,
            "stem": csv_path.stem,
            "run_id_raw": parsed_name["run_id_raw"],
            "normalized_run_id": parsed_name["normalized_run_id"],
            "timestamp": parsed_name["timestamp"],
            "timestamp_iso": parsed_name["timestamp_iso"],
            "acquisition_tag": parsed_name["acquisition_tag"],
            "path": str(csv_path),
            "relative_path": relative_path,
            "pair_key": (
                f"position_{position_id}__run_{parsed_name['normalized_run_id']}"
                if position_id is not None and parsed_name["normalized_run_id"] is not None
                else None
            ),
            "header_valid": header == expected_header,
            "row_count": row_count,
            "channel_count": max(0, len(header) - 1),
        }
        records.append(record)

        for anomaly in file_anomalies:
            anomaly_records.append(
                {
                    "raw_file_id": raw_file_id,
                    "relative_path": relative_path,
                    **anomaly,
                }
            )

    dataset_index = pd.DataFrame(records)
    anomalies = pd.DataFrame(anomaly_records)

    if not dataset_index.empty and scan_row_count and "row_count" in dataset_index:
        median_row_count = dataset_index["row_count"].median()
        ratio = float(cfg.get("warn_row_count_deviation_ratio", 0.05))
        lower = median_row_count * (1.0 - ratio)
        upper = median_row_count * (1.0 + ratio)
        deviants = dataset_index[
            dataset_index["row_count"].notna()
            & ((dataset_index["row_count"] < lower) | (dataset_index["row_count"] > upper))
        ]
        if not deviants.empty:
            extra = deviants.apply(
                lambda row: {
                    "raw_file_id": row["raw_file_id"],
                    "relative_path": row["relative_path"],
                    "anomaly_type": "row_count_outlier",
                    "severity": "medium",
                    "message": f"Row count {row['row_count']} deviates from median {median_row_count:.0f}.",
                },
                axis=1,
            ).tolist()
            anomalies = pd.concat([anomalies, pd.DataFrame(extra)], ignore_index=True)

    if anomalies.empty:
        anomalies = pd.DataFrame(columns=["raw_file_id", "relative_path", "anomaly_type", "severity", "message"])

    return dataset_index, anomalies


def build_pairing_report(dataset_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    if dataset_index.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = dataset_index.groupby("pair_key", dropna=False)
    for pair_key, group in grouped:
        if pair_key is None or (isinstance(pair_key, float) and math.isnan(pair_key)):
            continue
        axial = group[group["orientation"] == "axial"]
        radial = group[group["orientation"] == "radial"]
        status = "paired"
        if len(axial) == 0:
            status = "missing_axial"
        elif len(radial) == 0:
            status = "missing_radial"
        elif len(axial) > 1 or len(radial) > 1:
            status = "ambiguous_pair"

        tags = sorted({tag for tag in group["acquisition_tag"].dropna().tolist()})
        tag_mismatch = len(tags) > 1
        rows.append(
            {
                "pair_key": pair_key,
                "position_id": group["position_id"].iloc[0],
                "normalized_run_id": group["normalized_run_id"].iloc[0],
                "axial_count": len(axial),
                "radial_count": len(radial),
                "axial_raw_file_ids": "|".join(axial["raw_file_id"].astype(str).tolist()),
                "radial_raw_file_ids": "|".join(radial["raw_file_id"].astype(str).tolist()),
                "axial_files": "|".join(axial["relative_path"].tolist()),
                "radial_files": "|".join(radial["relative_path"].tolist()),
                "acquisition_tags": "|".join(tags),
                "tag_mismatch": tag_mismatch,
                "status": status,
            }
        )

        if status != "paired":
            anomalies.append(
                {
                    "raw_file_id": "|".join(group["raw_file_id"].astype(str).tolist()),
                    "relative_path": "|".join(group["relative_path"].tolist()),
                    "anomaly_type": status,
                    "severity": "high" if status.startswith("missing") else "medium",
                    "message": f"Pair '{pair_key}' has axial_count={len(axial)} and radial_count={len(radial)}.",
                }
            )
        if tag_mismatch:
            anomalies.append(
                {
                    "raw_file_id": "|".join(group["raw_file_id"].astype(str).tolist()),
                    "relative_path": "|".join(group["relative_path"].tolist()),
                    "anomaly_type": "pair_acquisition_tag_mismatch",
                    "severity": "medium",
                    "message": f"Pair '{pair_key}' spans multiple acquisition tags: {tags}.",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(anomalies)


def build_acquisition_summary(dataset_index: pd.DataFrame) -> pd.DataFrame:
    if dataset_index.empty:
        return pd.DataFrame()
    return (
        dataset_index.fillna({"acquisition_tag": "MISSING"})
        .groupby(["acquisition_tag", "position_id", "orientation"])
        .size()
        .reset_index(name="file_count")
        .sort_values(["acquisition_tag", "position_id", "orientation"])
    )


def find_external_label_files(context: ProjectContext) -> list[Path]:
    candidates: list[Path] = []
    if context.label_path.exists():
        candidates.append(context.label_path)
    for pattern in ("*label*.csv", "*labels*.csv", "*metadata*.csv"):
        candidates.extend(context.data_root.rglob(pattern))
    deduped: dict[str, Path] = {str(path.resolve()): path.resolve() for path in candidates}
    return sorted(deduped.values())


def write_readiness_report(
    context: ProjectContext,
    dataset_index: pd.DataFrame,
    pairing_report: pd.DataFrame,
    anomalies: pd.DataFrame,
    acquisition_summary: pd.DataFrame,
    label_files: list[Path],
) -> None:
    total_csv = len(dataset_index)
    per_position_orientation = (
        dataset_index.groupby(["position_id", "orientation"]).size().reset_index(name="file_count")
        if not dataset_index.empty
        else pd.DataFrame(columns=["position_id", "orientation", "file_count"])
    )
    paired_success = int((pairing_report["status"] == "paired").sum()) if not pairing_report.empty else 0
    naming_anomalies = len(anomalies)
    independent_units = dataset_index["defect_unit_id"].dropna().nunique() if not dataset_index.empty else 0
    label_status = "FOUND" if label_files else "NOT FOUND"
    supervised_ready = bool(label_files) and context.label_path.exists()
    recommended_unit = (
        "physical defect unit at the position level (position_id), with run-level repeated measurements nested beneath it"
    )
    position_rows = (
        per_position_orientation[["position_id", "orientation", "file_count"]].values.tolist()
        if not per_position_orientation.empty
        else []
    )
    acquisition_rows = (
        acquisition_summary[["acquisition_tag", "position_id", "orientation", "file_count"]].values.tolist()
        if not acquisition_summary.empty
        else []
    )

    report = [
        "# Data Readiness Report",
        "",
        "## Prompt-required answers",
        f"1. Total CSV count: **{total_csv}**",
        "2. File counts by position and orientation:",
        _to_markdown_table(["position_id", "orientation", "file_count"], position_rows)
        if position_rows
        else "No CSV files found.",
        f"3. Successfully paired axial-radial combinations: **{paired_success}**",
        f"4. Naming/path/header anomalies: **{naming_anomalies}**",
        f"5. External label file found: **{label_status}**",
        f"6. Recommended independent sample unit: **{recommended_unit}**",
        (
            "7. Suitable for supervised inversion now: **YES**"
            if supervised_ready
            else "7. Suitable for supervised inversion now: **NO**. Stay in weakly supervised / proxy mode until `metadata/defect_labels.csv` is populated."
        ),
        "",
        "## Reviewer-facing interpretation",
        f"- Distinct physical defect units currently observable from folder structure: **{independent_units}**.",
        "- Repeated runs and sliding windows must remain nested under the same physical defect unit during all splits.",
        "- Acquisition tags should not be silently merged when pairing; mismatches are reported explicitly.",
        "",
        "## Acquisition tag summary",
        _to_markdown_table(["acquisition_tag", "position_id", "orientation", "file_count"], acquisition_rows)
        if acquisition_rows
        else "No acquisition tags available.",
        "",
        "## Label discovery",
    ]
    if label_files:
        report.extend([f"- `{path}`" for path in label_files])
    else:
        report.append("- No external label file was discovered.")
    report.extend(
        [
            "",
            "## Mandatory caveats",
            "- Do not claim quantitative defect inversion until geometric labels are available and validated.",
            "- Do not count repeated runs or windows as independent samples.",
            "- Treat Stage C results as defect-state/proxy modeling unless the label file is provided.",
            "",
        ]
    )
    context.readiness_report_path.write_text("\n".join(report), encoding="utf-8")


def run_audit(context: ProjectContext) -> dict[str, Path]:
    dataset_index, anomalies = scan_dataset(context)
    pairing_report, pairing_anomalies = build_pairing_report(dataset_index)
    acquisition_summary = build_acquisition_summary(dataset_index)
    label_files = find_external_label_files(context)

    all_anomalies = pd.concat(
        [frame for frame in (anomalies, pairing_anomalies) if not frame.empty],
        ignore_index=True,
    ) if (not anomalies.empty or not pairing_anomalies.empty) else pd.DataFrame(
        columns=["raw_file_id", "relative_path", "anomaly_type", "severity", "message"]
    )

    dataset_index.sort_values(["position_id", "orientation", "normalized_run_id", "relative_path"]).to_csv(
        context.dataset_index_path, index=False, encoding="utf-8-sig"
    )
    pairing_report.sort_values(["position_id", "normalized_run_id"]).to_csv(
        context.pairing_report_path, index=False, encoding="utf-8-sig"
    )
    all_anomalies.sort_values(["severity", "relative_path", "anomaly_type"]).to_csv(
        context.anomaly_report_path, index=False, encoding="utf-8-sig"
    )
    acquisition_summary.to_csv(
        context.reports_dir / "acquisition_tag_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_readiness_report(context, dataset_index, pairing_report, all_anomalies, acquisition_summary, label_files)
    return {
        "dataset_index": context.dataset_index_path,
        "pairing_report": context.pairing_report_path,
        "anomaly_report": context.anomaly_report_path,
        "acquisition_summary": context.reports_dir / "acquisition_tag_summary.csv",
        "readiness_report": context.readiness_report_path,
    }
