from __future__ import annotations

import argparse
import json
import warnings

from uq_pinn_mfl.config import load_context
from uq_pinn_mfl.data.labels import resolve_project_mode
from uq_pinn_mfl.data.preprocess import preprocess_dataset
from uq_pinn_mfl.simulation.realistic_mfl import generate_synthetic_dataset
from uq_pinn_mfl.training.pipelines import run_stage_a, run_stage_b, run_stage_c, run_stage_d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reviewer-constrained UQ-PINN + MFL research pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Run Stage A audit and readiness reports.")

    preprocess = subparsers.add_parser("preprocess", help="Cache CSV signals and build window manifests.")
    preprocess.add_argument("--rebuild", action="store_true")

    stage_b = subparsers.add_parser("stage-b", help="Run denoising baseline or ResUNet.")
    stage_b.add_argument("--model", default="resunet1d", choices=["resunet1d", "moving_average"])
    stage_b.add_argument("--rebuild", action="store_true")

    stage_c = subparsers.add_parser("stage-c", help="Run weakly supervised / proxy surrogate training.")
    stage_c.add_argument("--variant", default="uq_pinn", choices=["blackbox", "deterministic_pinn", "uq_pinn"])
    stage_c.add_argument("--rebuild", action="store_true")

    stage_d = subparsers.add_parser("stage-d", help="Run supervised inversion if labels are available.")
    stage_d.add_argument("--variant", default="uq_pinn", choices=["blackbox", "deterministic_pinn", "uq_pinn"])
    stage_d.add_argument("--rebuild", action="store_true")
    stage_d.add_argument("--lopo", action="store_true", help="Run leave-one-position-out batch evaluation.")

    simulate = subparsers.add_parser("simulate", help="Generate synthetic MFL CSV signals aligned with the real dataset shape.")
    simulate.add_argument("--mode", default=None, choices=["benchmark", "match_real_layout"])
    simulate.add_argument("--dataset-name", default=None)

    subparsers.add_parser("mode", help="Report whether the project is in proxy or supervised mode.")
    return parser


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*CUDA capability sm_120 is not compatible with the current PyTorch installation.*",
    )
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.cuda")
    parser = build_parser()
    args = parser.parse_args()
    context = load_context(args.config)

    try:
        if args.command == "audit":
            payload = run_stage_a(context)
        elif args.command == "preprocess":
            payload = {key: str(value) for key, value in preprocess_dataset(context, rebuild=args.rebuild).items()}
        elif args.command == "stage-b":
            payload = run_stage_b(context, model_name=args.model, rebuild=args.rebuild)
        elif args.command == "stage-c":
            payload = run_stage_c(context, variant=args.variant, rebuild=args.rebuild)
        elif args.command == "stage-d":
            payload = run_stage_d(context, variant=args.variant, rebuild=args.rebuild, lopo=args.lopo)
        elif args.command == "simulate":
            payload = generate_synthetic_dataset(context, mode=args.mode, dataset_name=args.dataset_name)
        else:
            mode, mode_warnings = resolve_project_mode(context)
            payload = {"mode": mode, "warnings": mode_warnings}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as exc:
        error_payload = {"status": "error", "command": args.command, "message": str(exc)}
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
