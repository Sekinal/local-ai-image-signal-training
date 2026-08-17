from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .attacks import AttackCase
from .inference import _ImageDataset
from .io import read_jsonl, sha256_file


def run_onnx_inference(
    rows: list[dict[str, Any]],
    output: str | Path,
    model_path: str | Path,
    model_id: str,
    model_revision: str,
    attacks: list[AttackCase],
    batch_size: int,
    workers: int,
    device: str,
    input_size: int = 384,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    from torch.utils.data import DataLoader

    available = set(ort.get_available_providers())
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if provider not in available:
        raise RuntimeError(f"requested {provider}, available providers are {sorted(available)}")
    model_path = Path(model_path)
    fingerprint = sha256_file(model_path)
    session = ort.InferenceSession(
        str(model_path), providers=[provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider]
    )
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if len(input_meta) != 1 or not output_meta:
        raise ValueError(f"expected one ONNX input and at least one output, got {len(input_meta)} and {len(output_meta)}")
    input_name, output_name = input_meta[0].name, output_meta[0].name
    resize_short_edge = 440 if input_size == 384 else 256
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if output.exists():
        existing = {(row["sample_id"], row["view"]): row for row in read_jsonl(output)}
    temporary = output.with_suffix(output.suffix + ".tmp")
    started = time.monotonic()
    written = 0
    failures = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in existing.values():
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        for attack in attacks:
            pending = [row for row in rows if (row["sample_id"], attack.attack_id) not in existing]
            dataset = _ImageDataset(pending, attack, input_size, rows)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(max(0, workers), os.cpu_count() or 1),
                pin_memory=device == "cuda",
                persistent_workers=workers > 0 and bool(pending),
            )
            for images, indexes, errors in loader:
                logits = np.asarray(session.run([output_name], {input_name: images.numpy()})[0]).reshape(-1)
                if len(logits) != len(indexes):
                    raise ValueError(f"ONNX output batch has {len(logits)} values for {len(indexes)} inputs")
                for logit, index, error in zip(logits, indexes.tolist(), errors, strict=True):
                    source = pending[index]
                    failed = bool(error)
                    failures += int(failed)
                    record = {
                        "sample_id": source["sample_id"],
                        "view": attack.attack_id,
                        "raw_logit": None if failed else float(logit),
                        "error": error or None,
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "model_sha256": fingerprint,
                        "model_input_size": input_size,
                        "model_resize_short_edge": resize_short_edge,
                        **attack.metadata(),
                    }
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    written += 1
                handle.flush()
    os.replace(temporary, output)
    elapsed = time.monotonic() - started
    return {
        "rows": len(rows),
        "predictions_written": written,
        "resumed_predictions": len(existing),
        "failures": failures,
        "views": [attack.attack_id for attack in attacks],
        "seconds": elapsed,
        "predictions_per_second": written / elapsed if elapsed else None,
        "provider": provider,
        "model_sha256": fingerprint,
        "input": {"name": input_name, "shape": input_meta[0].shape, "type": input_meta[0].type},
        "output": {"name": output_name, "shape": output_meta[0].shape, "type": output_meta[0].type},
    }
