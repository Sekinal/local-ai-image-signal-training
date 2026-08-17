#!/usr/bin/env python3
"""Generate a pinned, train-only cohort with official Hugging Face models.

The model allow-list is intentionally narrow: official foundation-model repositories,
public/ungated weights, and Apache-2.0 licensing.  Each invocation is resumable and
records the exact model revision, runtime, prompt, seed, inference parameters, image
hash, and perceptual hash.  No Hugging Face credential is required or consumed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import HfApi

from aiblink_validation.hashing import image_metadata_and_phash
from aiblink_validation.io import atomic_json, sha256_file, write_manifest


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "flux2-klein-4b": {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "license": "Apache-2.0",
        "created_at": "2026-01-14",
        "precision": "bfloat16",
        "parameters": {"height": 1024, "width": 1024, "num_inference_steps": 4, "guidance_scale": 1.0},
    },
    "qwen-image-2512": {
        "model_id": "Qwen/Qwen-Image-2512",
        "revision": "25468b98e3276ca6700de15c6628e51b7de54a26",
        "license": "Apache-2.0",
        "created_at": "2025-12-30",
        "precision": "bitsandbytes-nf4-bfloat16",
        "parameters": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 50,
            "true_cfg_scale": 4.0,
            "negative_prompt": (
                "low resolution, low quality, deformed anatomy, malformed fingers, oversaturated, "
                "waxy skin, over-smoothed face, chaotic composition, blurry or distorted text"
            ),
        },
    },
    "z-image-turbo": {
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
        "license": "Apache-2.0",
        "created_at": "2025-11-25",
        "precision": "bfloat16",
        "parameters": {"height": 1024, "width": 1024, "num_inference_steps": 9, "guidance_scale": 0.0},
    },
    "sana-sprint-1.6b": {
        "model_id": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        "revision": "19683c58b7ea290e55cedd8950ae1d86ada7ef96",
        "license": "Apache-2.0",
        "created_at": "2025-03-21",
        "precision": "bfloat16",
        "parameters": {"height": 1024, "width": 1024, "num_inference_steps": 2},
    },
}

PROMPTS = (
    "An unposed smartphone photograph of three commuters sheltering beneath a narrow awning during heavy rain at blue hour, wet pavement, mixed storefront light, slight motion blur and imperfect framing",
    "A documentary photograph inside a neighborhood repair shop, an older mechanic fixing a small radio at a crowded workbench, ordinary fluorescent light, realistic tools, dust and worn surfaces",
    "A low-light camera-trap image of a fox crossing a muddy woodland path at night, infrared illumination, timestamp overlay, foreground branches and compression noise",
    "A casual family snapshot of people preparing dumplings in a compact lived-in kitchen, uneven indoor lighting, flour on the table, natural faces and partially occluded hands",
    "A wide real-estate listing photograph of a modest apartment living room, mismatched furniture, window glare, wide-angle distortion, visible cables and ordinary household clutter",
    "A press-style photograph of an amateur football match on an overcast afternoon, distant players, muddy field, spectators behind a fence and slightly missed telephoto focus",
    "A close phone photograph of a folded grocery receipt on a scratched wooden table, varied legible item text, hard side shadow, paper wrinkles and perspective skew",
    "A foggy coastal landscape captured on an older compact camera, waves striking dark rocks, muted color, lens droplets, tilted horizon and distant warning sign",
    "A crowded outdoor produce market immediately after rain, handwritten price cards, tarps, stacked crates, realistic shoppers, reflections and visual clutter",
    "A deliberately mediocre online marketplace photo of a used bicycle leaning against a garage wall, awkward crop, hard flash, uneven white balance and oil stains",
    "A microscope image of differently shaped stained cells on a glass slide, uneven illumination, dust specks, realistic optical aberration, scale bar and two small annotation arrows",
    "A security-camera still from a quiet warehouse aisle, high corner viewpoint, monochrome night mode, timestamp, lens distortion, low bitrate noise and a partially blocked view",
    "A scanned color family photograph from the late 1980s, people gathered around a birthday cake, faded dyes, soft focus, dust, crease marks and a handwritten date on the border",
    "A smartphone panorama of a busy city intersection at sunset, duplicated pedestrian caused by stitching, changing exposure, traffic, signage and mild rolling-shutter distortion",
    "A detailed stylized digital illustration of a dense floating street market above a futuristic city, layered characters, varied handmade signs, painterly texture and humid atmosphere",
    "A physically based 3D render of inexpensive translucent headphones in opened retail packaging, small printed legends, fingerprints, plastic scuffs and a neutral studio sweep",
    "An ordinary phone photo from the rear seat of a taxi during evening traffic, windshield reflections, navigation screen glow, brake lights, compression and slight hand shake",
    "A community notice board in a library lobby covered with overlapping flyers, event dates, tear-off tabs, pushpins, glare and partially obscured readable text",
    "A close documentary photograph of a baker arranging bread before sunrise, flour dust, stainless steel trays, mixed fluorescent and window light, natural skin and worn clothing",
    "A dashcam frame on a wet rural road at twilight, an approaching truck, windshield streaks, headlight bloom, block compression and a small date overlay",
    "A phone photograph of a television showing a colorful weather forecast, dim living-room reflections, screen moire, off-axis perspective and remote controls on a cluttered table",
    "A school science worksheet photographed on a desk, hand-drawn circuit diagram, pencil calculations, eraser crumbs, ruled annotations, fold shadows and legible labels",
    "A handheld photograph of a small live-music venue from behind the audience, raised phones, colored spotlights, smoke, clipped highlights and realistic motion blur",
    "An overhead aerial photograph of irregular farmland after flooding, muddy water, narrow roads, scattered buildings, cloud shadows and varying seasonal vegetation",
    "A candid photograph in a laundromat late at night, several people waiting, rows of machines, detergent bottles, harsh ceiling light and reflections in scratched windows",
    "A close wildlife photograph of a wet sparrow on a rusty railing, fine feather detail, overcast daylight, shallow depth of field and an untidy urban background",
    "A hospital corridor photographed with an older phone, empty wheelchairs, wayfinding signs, mixed color temperature, polished-floor reflections and digital noise",
    "A small restaurant menu board photographed through a window, multilingual dish names and prices, marker lettering, glare, stickers, partial occlusion and people inside",
    "A tabletop product photograph of a budget electric kettle removed from dented packaging, instruction leaflet, cable ties, fingerprints, hard flash and kitchen clutter",
    "A construction progress photograph of an unfinished concrete interior, exposed conduit, safety markings, dust, temporary lights, workers in the distance and strong perspective",
    "A macro photograph of an old circuit board under a desk lamp, solder joints, corrosion, dust fibers, component labels, uneven focus and realistic sensor noise",
    "A ferry-terminal security camera still during rain, high viewpoint, passengers with luggage, reflective floor, timestamp, muted color and low-bitrate artifacts",
    "A scanned newspaper clipping with a local sports photograph, halftone dots, yellowed paper, uneven crop, fold crease, handwritten circle and partly readable caption",
    "A casual phone portrait of an elderly gardener beside a greenhouse, natural wrinkles, muddy gloves, windblown hair, bright cloudy light and ordinary background clutter",
    "A crowded commuter train interior photographed discreetly at morning rush hour, varied passengers, poles and advertisements, mixed lighting and imperfect autofocus",
    "A realistic online meeting screenshot with twelve distinct participants, varied webcams, names, mute icons, screen-share thumbnail, notification banner and compression",
    "A messy spreadsheet dashboard shown on an office monitor, small column labels, charts, conditional formatting, warning dialog, glare, moire and realistic interface spacing",
    "A photographed hand-drawn street map with route arrows, crossed-out notes, coffee stain, colored highlighter, folded paper and a metal ruler at the edge",
    "A medical X-ray viewer photographed in a dark reading room, two chest images, interface panels, faint reflections, exposure variation and no patient-identifying information",
    "A pathology slide image with irregular tissue staining, small bubbles, edge vignetting, scale marker, two annotation boxes and realistic microscope softness",
    "A natural-history museum display case photographed by a visitor, mineral specimens, small labels, glass reflections, uneven spotlights and people reflected behind the camera",
    "A satellite-style image of a coastal port with containers, ships, cranes, road markings, water haze, cloud wisps and subtle sharpening artifacts",
    "A watercolor-and-ink illustration of a crowded neighborhood café, varied patrons, handwritten wall menus, loose linework, paper grain and muted afternoon color",
    "A hand-painted animation frame of a rural railway platform in summer, several distinct travelers, luggage, weathered signs, heat haze and expressive background detail",
    "A low-poly 3D game environment showing an abandoned service station at dusk, modular props, worn signage, puddles, volumetric fog and physically plausible lighting",
    "A clean technical cutaway diagram of a compact heat pump, numbered components, arrows, small legends, colored fluid paths, subtle print texture and precise alignment",
    "A collage-style social media post about a lost pet, phone photograph, bold headline, map fragment, contact strip, reaction icons, crop marks and recompression artifacts",
    "An editorial food photograph of a half-eaten noodle dish on a plastic table, chopsticks, spilled sauce, napkins, mixed street lighting and busy out-of-focus background",
)

PROMPT_VARIANTS = (
    "Keep the scene plausible and restrained, with ordinary imperfections rather than polished advertising aesthetics.",
    "Emphasize fine local texture, realistic spatial relationships, and small incidental details without adding a watermark.",
    "Use an asymmetrical composition with believable occlusion, edge clutter, and nonuniform illumination.",
    "Preserve mundane materials and physically consistent light while allowing minor capture or rendering artifacts.",
    "Favor documentary specificity, varied object scale, and an imperfect but coherent background.",
    "Include subtle noise, texture variation, and natural wear appropriate to the requested medium.",
)

ASPECT_SIZES = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jobs(slug: str, per_model: int, seed: int) -> list[dict[str, Any]]:
    config = MODEL_CONFIGS[slug]
    jobs = []
    model_index = list(MODEL_CONFIGS).index(slug)
    for index in range(per_model):
        image_seed = seed + model_index * 100_003 + index * 997
        if index < 8:
            # Preserve the exact pilot jobs even as the larger prompt matrix evolves.
            prompt_index = (model_index * 3 + index * 5) % 16
            prompt = PROMPTS[prompt_index]
            parameters = dict(config["parameters"])
            payload = f"{config['model_id']}\0{config['revision']}\0{image_seed}\0{prompt}"
        else:
            prompt_index = (model_index * 11 + index * 17) % len(PROMPTS)
            variant_index = (model_index + index // len(PROMPTS)) % len(PROMPT_VARIANTS)
            prompt = f"{PROMPTS[prompt_index]} {PROMPT_VARIANTS[variant_index]}"
            width, height = ASPECT_SIZES[(model_index + index) % len(ASPECT_SIZES)]
            parameters = {**config["parameters"], "width": width, "height": height}
            payload = (
                f"{config['model_id']}\0{config['revision']}\0{image_seed}\0"
                f"{width}x{height}\0{prompt}"
            )
        jobs.append(
            {
                "job_id": hashlib.sha256(payload.encode()).hexdigest()[:24],
                "index": index,
                "prompt": prompt,
                "seed": image_seed,
                "parameters": parameters,
            }
        )
    return jobs


def _validate_model(slug: str) -> None:
    config = MODEL_CONFIGS[slug]
    info = HfApi(token=False).model_info(config["model_id"], revision=config["revision"])
    license_tags = {tag for tag in info.tags if tag.startswith("license:")}
    if info.gated or info.private:
        raise RuntimeError(f"{config['model_id']} is no longer public and ungated")
    if "license:apache-2.0" not in license_tags:
        raise RuntimeError(f"{config['model_id']} no longer reports Apache-2.0: {sorted(license_tags)}")
    if info.sha != config["revision"]:
        raise RuntimeError(f"revision mismatch for {config['model_id']}: {info.sha}")


def _load_pipeline(slug: str) -> DiffusionPipeline:
    config = MODEL_CONFIGS[slug]
    kwargs: dict[str, Any] = {
        "revision": config["revision"],
        "torch_dtype": torch.bfloat16,
        "token": False,
        "low_cpu_mem_usage": True,
        "device_map": "cuda",
    }
    if slug == "qwen-image-2512":
        from diffusers.quantizers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
    return DiffusionPipeline.from_pretrained(config["model_id"], **kwargs)


def _manifest_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record.get("error"):
            continue
        model_id = record["model_id"]
        revision = record["revision"]
        digest = record["content_sha256"]
        rows.append(
            {
                "sample_id": hashlib.sha256(f"HuggingFace-recent\0{model_id}\0{digest}".encode()).hexdigest()[:24],
                "path": record["path"],
                "label": 1,
                "dataset": "HuggingFace-recent-generators",
                "source": f"HuggingFace:{model_id}",
                "group_id": f"HuggingFace:{model_id}@{revision}",
                "role": "train",
                "content_sha256": digest,
                "phash": record["phash"],
                "width": record["width"],
                "height": record["height"],
                "mime": record["mime"],
                "license": record["license"],
                "generator_revision": revision,
                "generator_precision": record["precision"],
                "generation_seed": record["seed"],
            }
        )
    return sorted(rows, key=lambda row: (row["source"], row["sample_id"]))


def _write_state(output: Path, ledger: Path, args: argparse.Namespace) -> None:
    records = _read_jsonl(ledger)
    successes = [record for record in records if not record.get("error")]
    manifest = output / "manifest.csv"
    if successes:
        write_manifest(manifest, _manifest_rows(successes))
    summary = {
        "protocol": "aiblink-huggingface-acquisition/0.2.0",
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_seed": args.seed,
        "requested_per_model": args.per_model,
        "requested_models": args.models,
        "successes": len(successes),
        "failures": sum(bool(record.get("error")) for record in records),
        "by_model": dict(Counter(record["model_id"] for record in successes)),
        "ledger_sha256": sha256_file(ledger) if ledger.exists() else None,
        "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
        "role": "train",
        "model_policy": "official public ungated foundation checkpoints; Apache-2.0; nonparticipant",
        "credential_required": False,
        "credential_persisted": False,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    try:
        import diffusers
        import transformers

        summary["runtime"].update({"diffusers": diffusers.__version__, "transformers": transformers.__version__})
    except ImportError:
        pass
    atomic_json(output / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_CONFIGS), default=list(MODEL_CONFIGS))
    parser.add_argument("--per-model", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if args.per_model <= 0:
        raise SystemExit("--per-model must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "acquisition.jsonl"
    existing = {record["job_id"] for record in _read_jsonl(ledger) if not record.get("error")}
    invocation_failures = 0

    for slug in args.models:
        config = MODEL_CONFIGS[slug]
        _validate_model(slug)
        pending = [job for job in _jobs(slug, args.per_model, args.seed) if job["job_id"] not in existing]
        if not pending:
            print(f"{slug}: already complete", flush=True)
            continue
        print(f"{slug}: loading {config['model_id']}@{config['revision']}", flush=True)
        try:
            pipe = _load_pipeline(slug)
        except Exception as exc:
            invocation_failures += 1
            _append_jsonl(
                output / "errors.jsonl",
                {
                    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model_slug": slug,
                    "model_id": config["model_id"],
                    "revision": config["revision"],
                    "stage": "load",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"{slug}: load failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue

        for job in pending:
            try:
                generator = torch.Generator(device="cuda").manual_seed(job["seed"])
                image = pipe(prompt=job["prompt"], generator=generator, **job["parameters"]).images[0]
                directory = output / "images" / slug / "fake"
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"{job['job_id']}.png"
                temporary = target.with_suffix(".png.tmp")
                image.save(temporary, format="PNG", optimize=False)
                os.replace(temporary, target)
                width, height, image_format, phash = image_metadata_and_phash(target)
                digest = sha256_file(target)
                record = {
                    **job,
                    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model_slug": slug,
                    "model_id": config["model_id"],
                    "revision": config["revision"],
                    "model_created_at": config["created_at"],
                    "license": config["license"],
                    "precision": config["precision"],
                    "generation_parameters": job["parameters"],
                    "path": str(target),
                    "content_sha256": digest,
                    "phash": phash,
                    "width": width,
                    "height": height,
                    "mime": f"image/{(image_format or 'png').lower()}",
                    "label": 1,
                    "role": "train",
                    "provider": "Hugging Face Hub local inference",
                    "error": None,
                }
                _append_jsonl(ledger, record)
                existing.add(job["job_id"])
                print(f"{slug}: {job['index'] + 1}/{args.per_model} {target.name} {digest[:12]}", flush=True)
            except Exception as exc:
                invocation_failures += 1
                _append_jsonl(
                    output / "errors.jsonl",
                    {
                        **job,
                        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "model_slug": slug,
                        "model_id": config["model_id"],
                        "revision": config["revision"],
                        "stage": "generate",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                print(f"{slug}: generation failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        _write_state(output, ledger, args)

    _write_state(output, ledger, args)
    print((output / "summary.json").read_text(encoding="utf-8"), flush=True)
    return 1 if invocation_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
