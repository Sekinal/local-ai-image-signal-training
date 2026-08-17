from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .attacks import AttackCase
from .inference import _load_model, preprocess


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class _CleanDataset:
    def __init__(self, rows: list[dict[str, Any]], input_size: int):
        self.rows = rows
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        try:
            with Image.open(row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                normalized = torch.from_numpy(preprocess(image, self.input_size))
            return normalized, float(row["label"]), index, ""
        except Exception as exc:
            return torch.zeros(3, self.input_size, self.input_size), float(row["label"]), index, f"{type(exc).__name__}: {exc}"


def pgd_linf(
    model,
    clean_pixels,
    labels,
    epsilon: float,
    steps: int,
    restarts: int,
    mean,
    std,
):
    """Untargeted per-sample PGD that maximizes binary classification loss."""
    import torch
    import torch.nn.functional as functional

    epsilon = float(epsilon)
    step_size = max(epsilon / max(1, steps // 2), 0.25 / 255)
    best_pixels = clean_pixels.clone()
    best_loss = torch.full((len(clean_pixels),), -torch.inf, device=clean_pixels.device)
    lower, upper = torch.clamp(clean_pixels - epsilon, 0, 1), torch.clamp(clean_pixels + epsilon, 0, 1)
    for restart in range(restarts):
        if restart == 0:
            adversarial = clean_pixels.clone()
        else:
            adversarial = torch.clamp(clean_pixels + torch.empty_like(clean_pixels).uniform_(-epsilon, epsilon), 0, 1)
        for _ in range(steps):
            adversarial.requires_grad_(True)
            logits = model((adversarial - mean) / std).squeeze(-1)
            loss = functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = torch.clamp(adversarial.detach() + step_size * gradient.sign(), lower, upper)
        with torch.no_grad():
            logits = model((adversarial - mean) / std).squeeze(-1)
            losses = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            improve = losses > best_loss
            best_loss[improve] = losses[improve]
            best_pixels[improve] = adversarial[improve]
    return best_pixels.detach()


def run_whitebox(
    rows: list[dict[str, Any]],
    output: str | Path,
    model_id: str,
    revision: str | None,
    epsilons: list[int],
    steps: int,
    restarts: int,
    batch_size: int,
    workers: int,
    device: str,
    attack_cases: list[AttackCase] | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("white-box validation currently requires CUDA")
    model, input_size, snapshot = _load_model(model_id, revision, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = _CleanDataset(rows, input_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(max(0, workers), os.cpu_count() or 1),
        pin_memory=True,
    )
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    records = []
    failures = 0
    started = time.monotonic()
    clean_case = AttackCase("clean", "clean", "control", ())
    if attack_cases is None:
        whitebox_cases = [
            AttackCase(
                f"pgd_linf_{epsilon}_255",
                "whitebox",
                "high",
                ({"op": "whitebox_pgd", "epsilon": epsilon, "steps": steps, "restarts": restarts},),
            )
            for epsilon in epsilons
        ]
    else:
        clean_matches = [case for case in attack_cases if case.attack_id == "clean"]
        if len(clean_matches) != 1:
            raise ValueError("white-box attack profile must contain exactly one clean control")
        clean_case = clean_matches[0]
        whitebox_cases = [case for case in attack_cases if case.attack_id != "clean"]
        for case in whitebox_cases:
            if len(case.operations) != 1 or case.operations[0].get("op") != "whitebox_pgd":
                raise ValueError(f"white-box profile case {case.attack_id!r} must contain one whitebox_pgd operation")
    for normalized, labels, indexes, errors in loader:
        keep = torch.tensor([not bool(error) for error in errors], dtype=torch.bool)
        failures += int((~keep).sum())
        if not keep.any():
            continue
        normalized = normalized[keep].to(device, non_blocking=True)
        labels_device = labels[keep].float().to(device)
        kept_indexes = indexes[keep].tolist()
        clean_pixels = torch.clamp(normalized * std + mean, 0, 1)
        with torch.no_grad():
            clean_logits = model(normalized).squeeze(-1).float().cpu().numpy()
        for index, logit in zip(kept_indexes, clean_logits, strict=True):
            records.append(
                {
                    "sample_id": rows[index]["sample_id"], "view": "clean", "attack_id": "clean",
                    **clean_case.metadata(), "raw_logit": float(logit),
                    "model_id": model_id, "model_revision": revision or Path(snapshot).name, "error": None,
                }
            )
        for attack in whitebox_cases:
            operation = attack.operations[0]
            epsilon_units = int(operation["epsilon"])
            attack_steps = int(operation["steps"])
            attack_restarts = int(operation["restarts"])
            adversarial = pgd_linf(
                model, clean_pixels, labels_device, epsilon_units / 255, attack_steps, attack_restarts, mean, std
            )
            with torch.no_grad():
                logits = model((adversarial - mean) / std).squeeze(-1).float().cpu().numpy()
            for index, logit in zip(kept_indexes, logits, strict=True):
                records.append(
                    {
                        "sample_id": rows[index]["sample_id"], "view": attack.attack_id,
                        **attack.metadata(), "raw_logit": float(logit),
                        "epsilon": epsilon_units / 255, "steps": attack_steps, "restarts": attack_restarts,
                        "model_id": model_id, "model_revision": revision or Path(snapshot).name, "error": None,
                    }
                )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, output)
    elapsed = time.monotonic() - started
    return {
        "rows": len(rows), "predictions": len(records), "failures": failures,
        "attacks": [case.metadata() for case in [clean_case, *whitebox_cases]], "seconds": elapsed,
    }
