from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_manifest
from .io import atomic_json, read_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aiblink-val", description="AI Blink validation pipeline")
    subcommands = parser.add_subparsers(dest="command", required=True)

    audit = subcommands.add_parser("audit", help="audit a manifest for leakage and structural errors")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--phash-distance", type=int, default=4)
    audit.add_argument("--skip-file-check", action="store_true")

    manifest = subcommands.add_parser("manifest-from-folders", help="hash images and build a manifest")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--role", choices=("train", "calibration", "test"), required=True)
    manifest.add_argument("--layout", choices=("generator/label", "label/generator"), required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--workers", type=int, default=4)

    inference = subcommands.add_parser("infer", help="run Community Forensics and write raw logits")
    inference.add_argument("--manifest", required=True)
    inference.add_argument("--role", default="calibration,test", help="comma-separated roles")
    inference.add_argument("--model", default="OwensLab/commfor-model-384")
    inference.add_argument("--revision")
    inference.add_argument("--views", default="clean,web,hard")
    inference.add_argument("--attack-config", help="YAML attack registry; replaces --views")
    inference.add_argument("--profile", default="smoke", help="profile within --attack-config")
    inference.add_argument("--batch-size", type=int, default=96)
    inference.add_argument("--workers", type=int, default=4)
    inference.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    inference.add_argument("--out", required=True)

    report = subcommands.add_parser("report", help="calibrate on calibration groups and report on test")
    report.add_argument("--manifest", required=True)
    report.add_argument("--predictions", required=True)
    report.add_argument("--config", required=True)
    report.add_argument("--out", required=True)

    comparison = subcommands.add_parser("compare", help="paired comparison of two calibrated test ledgers")
    comparison.add_argument("--champion", required=True)
    comparison.add_argument("--challenger", required=True)
    comparison.add_argument("--threshold", type=float, default=0.65)
    comparison.add_argument("--replicates", type=int, default=2000)
    comparison.add_argument("--seed", type=int, default=323)
    comparison.add_argument("--out", required=True)

    redteam = subcommands.add_parser("redteam-report", help="score a frozen calibrator against an attack profile")
    redteam.add_argument("--manifest", required=True)
    redteam.add_argument("--predictions", required=True)
    redteam.add_argument("--calibration", required=True)
    redteam.add_argument("--attack-config", required=True)
    redteam.add_argument("--profile", default="smoke")
    redteam.add_argument("--threshold", type=float, default=0.65)
    redteam.add_argument("--replicates", type=int, default=2000)
    redteam.add_argument("--seed", type=int, default=323)
    redteam.add_argument("--out", required=True)

    leakage = subcommands.add_parser("metadata-audit", help="cross-validated metadata-only leakage probe")
    leakage.add_argument("--manifest", required=True)
    leakage.add_argument("--role", default="train,calibration,test")
    leakage.add_argument("--folds", type=int, default=5)
    leakage.add_argument("--seed", type=int, default=323)
    leakage.add_argument("--out", required=True)

    whitebox = subcommands.add_parser("whitebox-eval", help="exact-model L-infinity PGD evaluation")
    whitebox.add_argument("--manifest", required=True)
    whitebox.add_argument("--role", default="test")
    whitebox.add_argument("--model", default="OwensLab/commfor-model-384")
    whitebox.add_argument("--revision")
    whitebox.add_argument("--epsilons", default="1,2,4", help="comma-separated values in /255")
    whitebox.add_argument("--steps", type=int, default=10)
    whitebox.add_argument("--restarts", type=int, default=1)
    whitebox.add_argument("--attack-config", help="use exact white-box settings from a YAML attack registry")
    whitebox.add_argument("--profile", default="adaptive", help="profile within --attack-config")
    whitebox.add_argument("--batch-size", type=int, default=16)
    whitebox.add_argument("--workers", type=int, default=4)
    whitebox.add_argument("--device", choices=("cuda",), default="cuda")
    whitebox.add_argument("--max-per-class", type=int)
    whitebox.add_argument("--out", required=True)

    dino_probe = subcommands.add_parser("dinov3-probe", help="fit a cross-fitted frozen DINOv3 linear probe")
    dino_probe.add_argument("--manifest", required=True)
    dino_probe.add_argument("--model", default="vit_small_patch16_dinov3.lvd1689m")
    dino_probe.add_argument("--repository", default="timm/vit_small_patch16_dinov3.lvd1689m")
    dino_probe.add_argument("--revision", default="3bf4720a82ec2066db88137180ff1f83a675cef0")
    dino_probe.add_argument("--input-size", type=int, default=384)
    dino_probe.add_argument("--views", default="clean,web,hard")
    dino_probe.add_argument("--ridge", type=float, default=1e-3)
    dino_probe.add_argument("--pooling", choices=("cls_mean_patch", "cls", "model_head", "spatial_mean"), default="cls_mean_patch")
    dino_probe.add_argument("--license", default="dinov3-license (evaluation only; not MIT-compatible for bounty redistribution)")
    dino_probe.add_argument("--batch-size", type=int, default=64)
    dino_probe.add_argument("--workers", type=int, default=4)
    dino_probe.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    dino_probe.add_argument("--out", required=True)

    backbone_probe = subcommands.add_parser("backbone-probe", help="fit a cross-fitted frozen timm backbone probe")
    backbone_probe.add_argument("--manifest", required=True)
    backbone_probe.add_argument("--model", required=True)
    backbone_probe.add_argument("--repository", required=True)
    backbone_probe.add_argument("--revision", required=True)
    backbone_probe.add_argument("--input-size", type=int, required=True)
    backbone_probe.add_argument("--pooling", choices=("cls_mean_patch", "cls", "model_head", "spatial_mean"), required=True)
    backbone_probe.add_argument("--license", required=True)
    backbone_probe.add_argument("--views", default="clean,web,hard")
    backbone_probe.add_argument("--ridge", type=float, default=1e-3)
    backbone_probe.add_argument("--batch-size", type=int, default=64)
    backbone_probe.add_argument("--workers", type=int, default=4)
    backbone_probe.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    backbone_probe.add_argument("--out", required=True)

    dino_infer = subcommands.add_parser("dinov3-infer", help="score an attack profile with a fitted DINOv3 probe")
    dino_infer.add_argument("--manifest", required=True)
    dino_infer.add_argument("--role", default="test")
    dino_infer.add_argument("--head", required=True)
    dino_infer.add_argument("--views", default="clean,web,hard")
    dino_infer.add_argument("--attack-config")
    dino_infer.add_argument("--profile", default="core")
    dino_infer.add_argument("--batch-size", type=int, default=64)
    dino_infer.add_argument("--workers", type=int, default=4)
    dino_infer.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    dino_infer.add_argument("--out", required=True)

    onnx_infer = subcommands.add_parser("onnx-infer", help="score a manifest with a local binary ONNX detector")
    onnx_infer.add_argument("--manifest", required=True)
    onnx_infer.add_argument("--role", default="calibration,test")
    onnx_infer.add_argument("--model-path", required=True)
    onnx_infer.add_argument("--model-id", required=True)
    onnx_infer.add_argument("--revision", required=True)
    onnx_infer.add_argument("--input-size", type=int, default=384)
    onnx_infer.add_argument("--views", default="clean,web,hard")
    onnx_infer.add_argument("--attack-config")
    onnx_infer.add_argument("--profile", default="core")
    onnx_infer.add_argument("--batch-size", type=int, default=96)
    onnx_infer.add_argument("--workers", type=int, default=4)
    onnx_infer.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    onnx_infer.add_argument("--out", required=True)

    train_pilot = subcommands.add_parser("train-pilot", help="equal-budget compiled fine-tuning pilot")
    train_pilot.add_argument("--manifest", required=True)
    train_pilot.add_argument("--candidate", required=True)
    train_pilot.add_argument("--steps", type=int, default=600)
    train_pilot.add_argument("--batch-size", type=int, default=96)
    train_pilot.add_argument("--lr", type=float, default=2e-5)
    train_pilot.add_argument("--head-multiplier", type=float, default=10.0)
    train_pilot.add_argument("--weight-decay", type=float, default=0.05)
    train_pilot.add_argument("--workers", type=int, default=4)
    train_pilot.add_argument("--seed", type=int, default=323)
    train_pilot.add_argument("--compile-mode", default="reduce-overhead")
    train_pilot.add_argument("--out", required=True)

    train_full = subcommands.add_parser("train-full", help="resumable source-balanced robust fine-tuning")
    train_full.add_argument("--manifest", required=True)
    train_full.add_argument("--initial-checkpoint", required=True)
    train_full.add_argument("--steps", type=int, default=4500)
    train_full.add_argument("--batch-size", type=int, default=96)
    train_full.add_argument("--lr", type=float, default=2e-5)
    train_full.add_argument("--head-multiplier", type=float, default=1.0)
    train_full.add_argument("--weight-decay", type=float, default=0.05)
    train_full.add_argument("--warmup-steps", type=int, default=300)
    train_full.add_argument("--workers", type=int, default=4)
    train_full.add_argument("--seed", type=int, default=323)
    train_full.add_argument("--compile-mode", default="reduce-overhead")
    train_full.add_argument("--validation-every", type=int, default=750)
    train_full.add_argument("--checkpoint-every", type=int, default=250)
    train_full.add_argument("--validation-cap", type=int, default=6000)
    train_full.add_argument("--validation-batch-size", type=int)
    train_full.add_argument("--ema-decay", type=float, default=0.999)
    train_full.add_argument("--source-sampling-exponent", type=float, default=0.5)
    train_full.add_argument("--resume")
    train_full.add_argument("--threshold", type=float, default=0.65)
    train_full.add_argument(
        "--augmentation-profile", choices=("web", "low-quality"), default="web"
    )
    train_full.add_argument("--out", required=True)

    score_trained = subcommands.add_parser("score-trained", help="score a fine-tuned tournament checkpoint")
    score_trained.add_argument("--manifest", required=True)
    score_trained.add_argument("--checkpoint", required=True)
    score_trained.add_argument("--role", default="calibration")
    score_trained.add_argument("--views", default="clean,web,hard")
    score_trained.add_argument("--attack-config", help="YAML attack registry; replaces --views")
    score_trained.add_argument("--profile", default="core", help="profile within --attack-config")
    score_trained.add_argument("--batch-size", type=int, default=128)
    score_trained.add_argument("--workers", type=int, default=4)
    score_trained.add_argument("--out", required=True)

    rank_tournament = subcommands.add_parser("rank-tournament", help="rank calibration-only tournament ledgers")
    rank_tournament.add_argument("--manifest", required=True)
    rank_tournament.add_argument("--predictions", nargs="+", required=True)
    rank_tournament.add_argument("--threshold", type=float, default=0.65)
    rank_tournament.add_argument("--out", required=True)

    export_trained = subcommands.add_parser("export-trained-onnx", help="export a trained checkpoint with verified FP16 ONNX weights")
    export_trained.add_argument("--checkpoint", required=True)
    export_trained.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    export_trained.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    export_trained.add_argument("--opset", type=int, default=17)
    export_trained.add_argument("--parity-batch", type=int, default=8)
    export_trained.add_argument("--parity-seed", type=int, default=323)
    export_trained.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        result = audit_manifest(
            read_manifest(args.manifest), args.phash_distance, check_files=not args.skip_file_check
        )
        atomic_json(args.out, result)
        print(json.dumps(result["summary"], indent=2))
        if not result["valid"]:
            print(f"audit failed with {len(result['issues'])} issue(s); see {args.out}", file=sys.stderr)
            return 2
        print(f"audit passed; wrote {args.out}")
        return 0
    if args.command == "manifest-from-folders":
        from .manifest import from_folders

        rows = from_folders(args.root, args.dataset, args.role, args.layout, args.out, args.workers)
        print(f"wrote {len(rows)} rows to {args.out}")
        return 0
    if args.command == "infer":
        attacks = None
        if args.attack_config:
            from .attacks import load_attack_profile

            attacks, _ = load_attack_profile(args.attack_config, args.profile)
        from .inference import run_inference

        roles = {item.strip() for item in args.role.split(",") if item.strip()}
        rows = [row for row in read_manifest(args.manifest) if row["role"] in roles]
        result = run_inference(
            rows,
            args.out,
            args.model,
            args.revision,
            [item.strip() for item in args.views.split(",") if item.strip()],
            args.batch_size,
            args.workers,
            args.device,
            attacks,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failures"] else 0
    if args.command == "report":
        from .report import generate_report

        result = generate_report(args.manifest, args.predictions, args.config, args.out)
        primary = result["evaluation"]["primary"]
        ci = result["evaluation"]["bootstrap_95"]["balanced_accuracy"]
        print(
            f"balanced_accuracy={primary['balanced_accuracy']:.4f} "
            f"95%_ci=[{ci['low']:.4f}, {ci['high']:.4f}] valid={result['valid']}"
        )
        print(f"wrote {Path(args.out).resolve()}")
        return 0 if result["valid"] else 3
    if args.command == "compare":
        from .comparison import compare

        result = compare(
            args.champion, args.challenger, args.out, args.threshold, args.replicates, args.seed
        )
        interval = result["delta_bootstrap_95"]["balanced_accuracy"]
        print(
            f"balanced_accuracy_delta={result['delta']['balanced_accuracy']:+.4f} "
            f"95%_ci=[{interval['low']:+.4f}, {interval['high']:+.4f}] "
            f"promotion_signal={result['promotion_signal']}"
        )
        return 0
    if args.command == "redteam-report":
        from .redteam import generate_redteam_report

        result = generate_redteam_report(
            args.manifest,
            args.predictions,
            args.calibration,
            args.attack_config,
            args.profile,
            args.out,
            args.threshold,
            args.replicates,
            args.seed,
        )
        robust = result["worst_variant_per_image"]["metrics"]
        print(
            f"worst_variant_ba={robust['balanced_accuracy']:.4f} "
            f"fake_any_attack={result['attack_success']['fake_any_attack']:.4f} valid={result['valid']}"
        )
        return 0 if result["valid"] else 4
    if args.command == "metadata-audit":
        from .leakage import metadata_leakage_audit

        roles = {role.strip() for role in args.role.split(",") if role.strip()}
        result = metadata_leakage_audit(args.manifest, args.out, roles, args.folds, args.seed)
        print(
            f"metadata_roc_auc={result['metrics']['roc_auc']:.4f} "
            f"group_cv_available={result['provenance_group_cv_available']} valid={result['valid']}"
        )
        return 0 if result["valid"] else 5
    if args.command == "whitebox-eval":
        from .whitebox import run_whitebox

        rows = [row for row in read_manifest(args.manifest) if row["role"] == args.role]
        if args.max_per_class:
            rows = [
                row
                for label in (0, 1)
                for row in [item for item in rows if int(item["label"]) == label][: args.max_per_class]
            ]
        attacks = None
        if args.attack_config:
            from .attacks import load_attack_profile

            attacks, _ = load_attack_profile(args.attack_config, args.profile)
        result = run_whitebox(
            rows, args.out, args.model, args.revision,
            [int(value) for value in args.epsilons.split(",")],
            args.steps, args.restarts, args.batch_size, args.workers, args.device,
            attacks,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failures"] else 0
    if args.command in {"dinov3-probe", "backbone-probe"}:
        from .attacks import legacy_case
        from .dinov3 import run_probe

        rows = read_manifest(args.manifest)
        calibration_rows = [row for row in rows if row["role"] == "calibration"]
        test_rows = [row for row in rows if row["role"] == "test"]
        attacks = [legacy_case(view.strip()) for view in args.views.split(",") if view.strip()]
        result = run_probe(
            calibration_rows, test_rows, attacks, args.out, args.model, args.repository, args.revision,
            args.input_size, args.ridge, args.batch_size, args.workers, args.device, args.pooling, args.license,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failures"] else 0
    if args.command == "dinov3-infer":
        from .attacks import legacy_case, load_attack_profile
        from .dinov3 import run_inference as run_dinov3_inference

        rows = [row for row in read_manifest(args.manifest) if row["role"] == args.role]
        if args.attack_config:
            attacks, _ = load_attack_profile(args.attack_config, args.profile)
        else:
            attacks = [legacy_case(view.strip()) for view in args.views.split(",") if view.strip()]
        result = run_dinov3_inference(
            rows, attacks, args.head, args.out, args.batch_size, args.workers, args.device
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failures"] else 0
    if args.command == "onnx-infer":
        from .attacks import legacy_case, load_attack_profile
        from .onnx_backend import run_onnx_inference

        roles = {role.strip() for role in args.role.split(",") if role.strip()}
        rows = [row for row in read_manifest(args.manifest) if row["role"] in roles]
        if args.attack_config:
            attacks, _ = load_attack_profile(args.attack_config, args.profile)
        else:
            attacks = [legacy_case(view.strip()) for view in args.views.split(",") if view.strip()]
        result = run_onnx_inference(
            rows, args.out, args.model_path, args.model_id, args.revision, attacks,
            args.batch_size, args.workers, args.device, args.input_size,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failures"] else 0
    if args.command == "train-pilot":
        from .training import train_pilot

        result = train_pilot(
            args.manifest, args.candidate, args.out, args.steps, args.batch_size, args.lr,
            args.head_multiplier, args.weight_decay, 0.10, args.workers, args.seed, args.compile_mode,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["decode_failures"] else 0
    if args.command == "train-full":
        from .training import train_full

        result = train_full(
            args.manifest, args.initial_checkpoint, args.out, args.steps, args.batch_size,
            args.lr, args.head_multiplier, args.weight_decay, args.warmup_steps, args.workers,
            args.seed, args.compile_mode, args.validation_every, args.checkpoint_every,
            args.validation_cap, args.ema_decay, args.source_sampling_exponent, args.resume,
            args.threshold, args.validation_batch_size, args.augmentation_profile,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["decode_failures"] else 0
    if args.command == "score-trained":
        from .attacks import load_attack_profile
        from .training import score_checkpoint

        attacks = None
        if args.attack_config:
            attacks, _ = load_attack_profile(args.attack_config, args.profile)

        result = score_checkpoint(
            args.manifest, args.checkpoint, args.out, args.role,
            tuple(view.strip() for view in args.views.split(",") if view.strip()),
            args.batch_size, args.workers, attacks,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failures"] else 0
    if args.command == "rank-tournament":
        from .training import rank_calibration

        result = rank_calibration(args.manifest, args.predictions, args.out, args.threshold)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "export-trained-onnx":
        from .export import export_trained_onnx

        result = export_trained_onnx(
            args.checkpoint, args.out, args.device, args.precision, args.opset, args.parity_batch,
            args.parity_seed,
        )
        print(json.dumps(result, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
