from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .io import atomic_json, sha256_file


def export_trained_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    precision: str = "fp16",
    opset: int = 17,
    parity_batch: int = 8,
    parity_seed: int = 323,
) -> dict[str, Any]:
    """Export an EMA tournament checkpoint with verified FP32 graph I/O."""
    import onnx
    import onnxruntime as ort
    import torch

    from .training import CandidateSpec, load_candidate

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"unsupported export precision {precision!r}")
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = CandidateSpec(**saved["candidate"])
    model, _ = load_candidate(spec, "cpu")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval().to(device)

    generator = torch.Generator(device="cpu").manual_seed(parity_seed)
    dummy_cpu = torch.randn(parity_batch, 3, spec.input_size, spec.input_size, generator=generator)
    dummy = dummy_cpu.to(device)
    with torch.inference_mode():
        reference = model(dummy).reshape(-1).float().cpu().numpy()

    class Float32IO(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module):
            super().__init__()
            self.wrapped = wrapped.half() if precision == "fp16" else wrapped.float()

        def forward(self, images):
            prepared = images.half() if precision == "fp16" else images.float()
            return self.wrapped(prepared).float()

    wrapper = Float32IO(model).eval()
    with torch.inference_mode():
        half_reference = wrapper(dummy).reshape(-1).float().cpu().numpy()

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    started = time.monotonic()
    torch.onnx.export(
        wrapper,
        dummy,
        temporary,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    graph = onnx.load(temporary, load_external_data=True)
    onnx.checker.check_model(graph)
    os.replace(temporary, output_path)

    available = set(ort.get_available_providers())
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if provider not in available:
        raise RuntimeError(f"requested {provider}, available providers are {sorted(available)}")
    session = ort.InferenceSession(
        str(output_path),
        providers=[provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider],
    )
    observed = np.asarray(session.run(["logits"], {"images": dummy_cpu.numpy()})[0]).reshape(-1)
    fp32_delta = np.abs(reference - observed)
    export_delta = np.abs(half_reference - observed)
    elapsed = time.monotonic() - started
    metadata = {
        "protocol": "aiblink-trained-onnx-export/0.1.0",
        "candidate": spec.candidate_id,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_sha256": sha256_file(output_path),
        "onnx_size_bytes": output_path.stat().st_size,
        "opset": opset,
        "precision": f"{precision} weights with fp32 input and output",
        "input": {"name": "images", "dtype": "float32", "shape": ["batch", 3, spec.input_size, spec.input_size]},
        "output": {"name": "logits", "dtype": "float32", "shape": ["batch", 1]},
        "parity": {
            "seed": parity_seed,
            "batch": parity_batch,
            "provider": provider,
            "fp32_checkpoint_vs_onnx_max_abs_logit": float(fp32_delta.max()),
            "fp32_checkpoint_vs_onnx_mean_abs_logit": float(fp32_delta.mean()),
            "export_precision_reference_vs_onnx_max_abs_logit": float(export_delta.max()),
            "export_precision_reference_vs_onnx_mean_abs_logit": float(export_delta.mean()),
        },
        "seconds": elapsed,
    }
    atomic_json(output_path.with_suffix(output_path.suffix + ".json"), metadata)
    return metadata
