#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from collections import Counter
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

API_ROOT = "https://openrouter.ai/api/v1"
MODEL_CONFIGS = (
    ("bytedance-seed/seedream-5-0-lite", {"resolution": "2K"}, 0.05),
    ("x-ai/grok-imagine-image-2.0", {"resolution": "1K", "quality": "low"}, 0.06),
    ("qwen/qwen-image-3", {"resolution": "1K"}, 0.05),
    ("microsoft/mai-image-2.5-pro", {}, 0.15),
    ("google/gemini-3.1-flash-lite-image", {"resolution": "1K"}, 0.08),
    ("openai/gpt-image-2", {"quality": "low"}, 0.10),
    ("sourceful/riverflow-v2.5-fast", {"resolution": "1K", "output_format": "jpeg"}, 0.04),
    ("recraft/recraft-v4.1", {}, 0.05),
    ("black-forest-labs/flux.2-klein-4b", {"output_format": "jpeg"}, 0.04),
)
ASPECT_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")
MODEL_ASPECT_RATIOS = {
    "recraft/recraft-v4.1": ("1:1", "4:3", "3:4", "16:9", "9:16"),
}
PROMPTS = (
    "An ordinary smartphone photo of two friends waiting at a rainy bus stop at dusk, natural skin texture, imperfect framing, wet pavement reflections",
    "A candid indoor photo of a family cooking dinner in a small lived-in kitchen, mixed warm and cool lighting, slight motion blur",
    "A low-light phone photo of a black dog running across a neighborhood park, digital zoom artifacts and uneven exposure",
    "A documentary photograph of a crowded outdoor produce market after rain, handwritten price signs, clutter, realistic people",
    "A tourist snapshot from inside a moving train, landscape visible through scratched glass, reflections and rolling-shutter distortion",
    "A close-up macro photograph of a chipped ceramic mug beside crumbs on an office desk, fluorescent lighting, shallow depth of field",
    "A real-estate listing style photograph of a modest apartment living room, wide-angle lens distortion, mixed furniture and window glare",
    "A press photograph of a local amateur football match on an overcast day, distant players, muddy field, imperfect telephoto focus",
    "A grocery receipt on a scratched wooden table photographed with a phone, legible varied text, folds, shadows, and perspective skew",
    "A screenshot-like image of a messy spreadsheet dashboard with charts, small labels, warning icons, and realistic UI spacing",
    "A photographed restaurant menu board with multilingual text, uneven chalk lettering, glare, partial occlusion, and people behind it",
    "A scientific field notebook page showing plant sketches, measurements, stains, arrows, and handwritten annotations",
    "A medical ultrasound screen photographed in a dim examination room, interface controls, subtle moire, and reflections",
    "A satellite-style overhead image of irregular farmland, roads, small buildings, clouds, and seasonal color variation",
    "A microscopy image of mixed cells with staining artifacts, dust, uneven illumination, scale bar, and annotation marks",
    "A product photo of inexpensive headphones in opened retail packaging on a kitchen counter, fingerprints and harsh phone flash",
    "A fashion editorial portrait outdoors in windy weather, natural pores, flyaway hair, fabric texture, and realistic background pedestrians",
    "A wildlife camera-trap image of a deer at night, infrared illumination, timestamp overlay, compression noise, and partial obstruction",
    "A foggy coastal landscape photographed on an older compact camera, muted colors, lens droplets, horizon slightly tilted",
    "An amateur concert photo from the back of a small venue, colored stage lights, raised phones, noise, blur, and clipped highlights",
    "A dashcam frame at night showing traffic, wet road, windshield reflections, headlight bloom, and blocky video compression",
    "A security-camera still of a quiet warehouse aisle, high viewpoint, timestamp, low bitrate, monochrome noise, and lens distortion",
    "A photorealistic plate of street food on a plastic table, mixed lighting, disposable utensils, sauce spills, and background clutter",
    "A deliberately mediocre marketplace listing photo of a used bicycle against a garage wall, awkward crop and uneven white balance",
    "A scanned vintage family photograph with faded color, dust, crease marks, soft focus, and handwritten date on the border",
    "A realistic architectural photo of a concrete stairwell, emergency signs, chipped paint, fluorescent lights, and strong perspective",
    "A cinematic but plausible frame of firefighters working beside a smoky roadside at dawn, documentary realism and atmospheric haze",
    "A children's crayon drawing taped to a refrigerator, uneven strokes, paper wrinkles, magnets, fingerprints, and kitchen background",
    "A detailed 3D product render of a translucent mechanical keyboard on a studio gradient, ray-traced materials and tiny printed legends",
    "A stylized digital illustration of a dense futuristic street market, varied signage, layered characters, painterly texture, and rain",
    "An anime-style classroom scene with many distinct students, detailed desks and posters, afternoon light, and hand-drawn line variation",
    "A procedural CGI landscape with eroded red rock arches, sparse vegetation, volumetric clouds, and physically based lighting",
    "A technical exploded-view diagram of a compact camera with numbered parts, clean vector lines, labels, and subtle paper texture",
    "A meme-like social media image combining a cat photo, bold caption text, reaction stickers, compression artifacts, and cropped UI chrome",
    "A realistic smartphone panorama of a busy city intersection with stitching errors, duplicated pedestrians, exposure changes, and traffic",
    "A photograph of a television displaying a colorful generated fantasy scene, room reflections, scan lines, moire, and off-axis perspective",
)


def request_json(url: str, key: str, payload: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, headers=headers, data=body, method="POST" if body else "GET")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def credits(key: str) -> dict:
    return request_json(f"{API_ROOT}/credits", key)["data"]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    records = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {record["request_id"] for record in records if not record.get("error")}


def plan(per_model: int, seed: int) -> list[dict]:
    jobs = []
    for model_index, (model, parameters, estimated_max) in enumerate(MODEL_CONFIGS):
        aspect_ratios = MODEL_ASPECT_RATIOS.get(model, ASPECT_RATIOS)
        for index in range(per_model):
            prompt_index = (model_index * per_model + index * 7) % len(PROMPTS)
            request_id = hashlib.sha256(f"{seed}\0{model}\0{index}\0{PROMPTS[prompt_index]}".encode()).hexdigest()[:24]
            jobs.append(
                {
                    "request_id": request_id,
                    "model": model,
                    "prompt": PROMPTS[prompt_index],
                    "aspect_ratio": aspect_ratios[(model_index + index) % len(aspect_ratios)],
                    "parameters": parameters,
                    "estimated_max_cost_usd": estimated_max,
                }
            )
    random.Random(seed).shuffle(jobs)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-usd", type=float, default=4.5)
    parser.add_argument("--per-model", type=int, default=10)
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.budget_usd <= 0 or args.per_model <= 0:
        raise SystemExit("budget and per-model count must be positive")
    jobs = plan(args.per_model, args.seed)
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "models": len(MODEL_CONFIGS), "budget_usd": args.budget_usd}, indent=2))
        return 0
    key = os.environ.get("OPENROUTER_TASK_KEY")
    if not key:
        raise SystemExit("OPENROUTER_TASK_KEY is required")
    from PIL import Image

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "acquisition.jsonl"
    completed = existing_ids(ledger)
    start_credit = credits(key)
    start_usage = float(start_credit["total_usage"])
    available = float(start_credit["total_credits"]) - start_usage
    if available <= 0:
        raise SystemExit("OpenRouter key has no remaining credits")
    tracked_cost = 0.0
    failures = 0
    written = 0
    for job in jobs:
        if job["request_id"] in completed:
            continue
        if tracked_cost + float(job["estimated_max_cost_usd"]) > min(args.budget_usd, available):
            break
        payload = {
            "model": job["model"],
            "prompt": job["prompt"],
            "n": 1,
            "aspect_ratio": job["aspect_ratio"],
            **job["parameters"],
        }
        result = None
        error = None
        for attempt in range(3):
            try:
                result = request_json(f"{API_ROOT}/images", key, payload)
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        if result is None or not result.get("data"):
            failures += 1
            append_jsonl(ledger, {**job, "created_at_utc": now, "error": error or "empty response"})
            continue
        usage_cost = float(result.get("usage", {}).get("cost") or job["estimated_max_cost_usd"])
        tracked_cost += usage_cost
        item = result["data"][0]
        raw = base64.b64decode(item["b64_json"], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            width, height = image.size
            image_format = (image.format or "PNG").lower()
        extension = {"jpeg": "jpg", "jpg": "jpg", "webp": "webp", "png": "png"}.get(image_format, "bin")
        digest = hashlib.sha256(raw).hexdigest()
        model_slug = job["model"].replace("/", "__")
        directory = output / "images" / model_slug / "fake"
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"{job['request_id']}-{digest[:12]}.{extension}"
        temporary = image_path.with_suffix(image_path.suffix + ".tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, image_path)
        append_jsonl(
            ledger,
            {
                **job,
                "created_at_utc": now,
                "model_returned": result.get("model"),
                "response_id": result.get("id"),
                "usage": result.get("usage", {}),
                "cost_usd": usage_cost,
                "path": str(image_path.resolve()),
                "content_sha256": digest,
                "width": width,
                "height": height,
                "format": image_format,
                "media_type": item.get("media_type"),
                "label": 1,
                "role": "train",
                "api": "OpenRouter Images API",
                "usage_rights_review_required": True,
                "error": None,
            },
        )
        written += 1
        if written % 5 == 0:
            current = credits(key)
            tracked_cost = max(tracked_cost, float(current["total_usage"]) - start_usage)
            print(json.dumps({"written": written, "failures": failures, "spent_usd": tracked_cost}), flush=True)
        if tracked_cost >= min(args.budget_usd, available):
            break
    end_credit = credits(key)
    actual_spend = float(end_credit["total_usage"]) - start_usage
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    successes = [record for record in records if not record.get("error")]
    summary = {
        "protocol": "aiblink-openrouter-acquisition/0.1.0",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "budget_usd": args.budget_usd,
        "start_available_credits_usd": available,
        "actual_spend_this_run_usd": actual_spend,
        "remaining_credits_usd": float(end_credit["total_credits"]) - float(end_credit["total_usage"]),
        "successes": len(successes),
        "failures": sum(bool(record.get("error")) for record in records),
        "by_model": dict(Counter(record["model"] for record in successes)),
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "role": "train",
        "usage_rights_review_required": True,
        "credential_persisted": False,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
