# Reproducing and continuing the August 2026 detector

This is the source snapshot used to train, evaluate, export, and promote the Community Forensics low-quality detector packaged by Local AI Image Signal v1.1.0.

## What is recoverable

| Layer | Status | Immutable reference |
|---|---|---|
| Training/evaluation source and tests | Public, complete | This repository |
| Initial and selected full checkpoints | Public, complete | `Thermostatic/community-forensics-low-quality-detector-2026-08@6fca3e7f4365363ee5c0fdb1a17d73917d54413d` |
| EMA safetensors and FP16 ONNX | Public, complete | Same revision |
| Calibrator, configs, reports, ledgers summarized in reports | Public, complete | Same revision |
| Exact 113,472-row manifest | Public metadata; paths and hashes included | Same revision, `reproduction/manifests/combined_manifest_original.csv.gz` |
| Portable path-free manifest inventory | Public, complete | Same revision, `combined_manifest_inventory.csv.gz` |
| 40,101-image frontier training component | Public image bytes | `Thermostatic/frontier-synthetic-images-2026@cb1df71639329a7e7b21a75bb6e838f3e38434cf` |
| Remaining 73,371 source images | Reacquire under upstream terms | Exact hashes, provenance, selection/materialization code, and composition are public; bytes are not republished here |
| OpenRouter-90 diagnostic bytes | Preserved owner-private pending rights review | `Thermostatic/openrouter-90-detector-diagnostic-private@3ba3b70d7da90bc95c5ac99b318d6a5d53e94737` |
| Chrome extension package | Public, complete | `Sekinal/local-ai-image-signal@v1.1.0` |

“Complete” above means the exact bytes and SHA-256 are available. Re-running GPU training is not promised to produce a byte-identical checkpoint: the recorded run used BF16, TF32, `torch.compile`, fused AdamW, multiple data-loader workers, and cuDNN benchmarking. Seeds and the stateless sampler make the sample plan reproducible, but GPU kernels and compiler versions may still introduce numerical drift. Use the released full checkpoint when exact inference or continued training matters.

## 1. Verify and fetch artifacts

Python 3.12 is the recorded interpreter. The exact important package versions are in `requirements-training-lock.txt`; the full host facts are in `environment.lock.json`.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training-lock.txt
python -m pip install -e .

# Deployment files, reports, and both manifest forms.
python scripts/fetch_reproduction_artifacts.py \
  --profiles deployment,data \
  --output reproduction/downloads

# Add the two 349 MB resumable checkpoints.
python scripts/fetch_reproduction_artifacts.py \
  --profiles resume \
  --output reproduction/downloads
```

Every download is pinned to a repository commit and checked against `reproduction/artifacts.json`. Private cohort restoration additionally requires the owner account's `HF_TOKEN` and `--profiles private`.

## 2. Restore or rehydrate the manifest

For the original server layout, the exact manifest is preferable because its path strings reproduce the checkpoint fingerprint:

```bash
mkdir -p /data/ai_blink/frontier_training
gzip -dc reproduction/downloads/data/combined_manifest_original.csv.gz \
  > /data/ai_blink/frontier_training/manifest.csv
sha256sum /data/ai_blink/frontier_training/manifest.csv
# expected: af629187fca31ab2f4f7f4c81ea3a761c13ea7cb8d4ee8a9bb9e7bd8e1eb6f36
```

Restore the image files at the paths recorded in that CSV. The public frontier component can be rebuilt with `scripts/build_frontier_public_dataset.py` and `scripts/prepare_frontier_training.py`. The legacy component is deterministically selected and materialized by `scripts/prepare_full_training.py` from OpenFake train/validation, Tiny-GenImage, SynthBuster, COCO train2017, and WikiArt. Upstream terms remain applicable.

If a different directory layout is required, rehydrate the portable inventory by exact content hash:

```bash
python scripts/rehydrate_manifest.py \
  --inventory reproduction/downloads/data/combined_manifest_inventory.csv.gz \
  --image-root /datasets/reacquired \
  --out data/manifest.csv
aiblink-val audit --manifest data/manifest.csv --out data/audit.json
```

That portable manifest is semantically identical but has different path strings, so its file SHA will differ from the original. It is suitable for new training. Exact optimizer resume uses the original manifest and `/data/ai_blink` layout because the checkpoint deliberately validates the manifest SHA.

## 3. Re-run the low-quality experiment

The original commands are preserved verbatim in `scripts/run_low_quality_experiment.sh`. With the original layout restored:

```bash
mkdir -p /root
ln -s "$PWD" /root/ai_blink
mkdir -p /data/ai_blink/final_commfor_20260816/run
cp reproduction/downloads/resume/frontier_initial_best.pt \
  /data/ai_blink/final_commfor_20260816/run/best.pt
bash scripts/run_low_quality_experiment.sh
```

The canary is 400 steps with batch 96 and seed 1323. It must pass its frozen gate before the 2,400-step, batch-96, seed-2323 scaled run begins. The scaled run uses peak LR `3e-6`, 120 warmup steps, AdamW weight decay 0.05, EMA 0.999, source-temperature exponent 0.35, and the class-symmetric low-quality augmentation profile.

To continue the released selected checkpoint rather than retrain, keep the exact manifest at its original path and pass the downloaded checkpoint to `train-full --resume`. Increase `--steps` beyond 2,400; all optimizer, scheduler, EMA, CPU RNG, and CUDA RNG state is present.

## 4. Re-run final evaluation and export

The exact orchestration is `scripts/run_low_quality_final_validation.sh`. It scores clean/web/hard, recent-HF, OpenRouter, eleven low-quality conditions, and the 33-condition red-team registry; fits the frozen calibrator; exports FP16 ONNX; performs full-ledger ONNX parity; and writes the promotion decision.

```bash
mkdir -p /data/ai_blink/low_quality_experiment_20260816/scaled
cp reproduction/downloads/resume/low_quality_selected_last.pt \
  /data/ai_blink/low_quality_experiment_20260816/scaled/last.pt
bash scripts/run_low_quality_final_validation.sh
```

The OpenRouter portion requires the owner-private diagnostic archive or may be skipped for a public-only reproduction. Skipping it means the protected OpenRouter recall gates cannot be independently reproduced. The recent-HF holdout is selected from the public frontier dataset and is identified by exact hashes in the published inventory.

Expected promotion-level results are in the immutable `reports/promotion_report.json`: low-quality macro balanced-accuracy gain `+0.159235`, macro fake-recall gain `+0.359913`, and ONNX threshold disagreement `0.230061%`. The red-team report remains invalid because small-composite and external-cohort gates failed.

## 5. Run tests

```bash
pytest
```

The release source passed all 35 Python tests on the training host. Artifact fetching and manifest rehydration have additional focused tests in `tests/test_reproduction_tools.py`.

## Security and competition boundary

No API keys, HF tokens, OpenRouter authorization headers, participant model weights, private competition images, or locked competition-test outputs are stored here. A legacy registry entry names Sieve as `disqualified-participant-model`; it was not used in the final training, selection, calibration, or export. The locked competition test was never opened.
