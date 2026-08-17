# Defensive red-team specification

The extension is a white-box detector running on a hostile page. Attackers can download the
ONNX graph, preprocessing, calibration parameters, and threshold, then test unlimited variants
offline. The page can also change what is displayed while inference is in flight.

Two invariants govern this work:

1. Every visible raster-bearing surface ends as `scored`, `pending`, `unsupported`,
   `permission-needed`, or `rate-limited`. Acquisition failure is never interpreted as `Real`.
2. A result is valid only for the exact, stable pixels and element generation to which it is
   attached. Stale or refetched-but-different pixels cannot receive a badge.

## P0 threats

| Family | Realistic attack | Primary validation |
|---|---|---|
| Multi-hop laundering | Resize → JPEG/WebP → screenshot → crop; repeated social-platform forwarding | One to six sequential hops, mixed codecs/resamplers, matched real/fake transforms |
| Partial synthesis | Generated/inpainted region inside a real photo, screenshot, meme, border, or collage | Synthetic area 1/2/5/10/25/50/75/100%; center, edge, and salient placement |
| Generative laundering | Low-strength img2img, restoration, relighting, denoising, face restoration, super-resolution | Multiple tool families and strengths; whole-frame and local edits |
| Physical recapture | Photograph display/print/projection; print-scan | Multiple phones/displays, angles, lighting, glare, moiré, and print paths |
| Best-of-N selection | Generate many seeds/prompts and publish the lowest detector score | N=10/100/1000; report evasion yield and worst-of-N per prompt family |
| White-box threshold attack | Optimize the exact shipped pipeline just below 0.65 | Multi-restart, EOT, localized and perceptual attacks; confirm in Chrome |
| Source mutation race | Reuse one node and swap real/fake pixels faster than inference | 50–200 ms swaps; zero stale result bindings |
| Render/refetch mismatch | CDN, cookie, service-worker, redirect, or expiring URL returns different bytes to extension | Pixel checksum must match rendered fixture; otherwise explicit unsupported/mismatch |
| Unenumerated surfaces | CSS backgrounds/pseudo-elements, canvas/WebGL, SVG, video, animation, shadow DOM, nested frames | Coverage accounting and known-pixel sentinel fixtures for every surface |
| Resource exhaustion | Thousands of tiny images, mutation storms, canvases, huge/pathological decodes | Bounded memory/queues; visible normal image still completes within latency SLO |
| UI spoofing | Fake badge, removed/covered badge, copied `data-*`, stale badge after crop/style change | Extension-owned popup/side panel remains authoritative |
| Asset/runtime corruption | Partial or modified model/config/WASM; OPFS quota; worker/GPU loss | Atomic verified activation, offline self-test, fail-closed recovery |

## Pixel and content matrix

Apply every ordinary transformation to both classes. Evaluate primitives and chains; transformed
copies remain clustered with their original image.

- Short edge: 64, 96, 128, 192, 256, 384, 512, 1024, 2048.
- Scale: 0.1, 0.25, 0.5, 0.7, 0.85, 1.5, 2, 4; nearest, area, bilinear,
  bicubic, and Lanczos; anisotropic and down-up cycles.
- JPEG: q100/95/90/85/75/60/50/30/20/10; 4:4:4 and 4:2:0; one/three/six
  generations. WebP/AVIF q90/70/50/30 and video H.264/H.265/AV1 frame extraction.
- Crop retained area: 5/10/25/50/75/90%; padding, borders, off-center crops, panoramas,
  rotations, shear, perspective, and CSS-visible crops.
- Gaussian blur σ0.3/0.7/1/2/3, motion blur, median blur, weak-to-strong denoise,
  sharpening, grain, Gaussian/Poisson/speckle noise.
- Gamma, saturation, contrast, temperature/tint, CLAHE, HDR/tone mapping, film filters,
  grayscale/CMYK, alpha compositing, ICC/P3/HDR, EXIF orientation, unusual bit depth.
- Text/logo/watermark/sticker coverage 5/10/25/50%, multiple locations and opacity.
- Generated screenshots, memes, UI, receipts, diagrams, AI-video frames, deliberate “bad
  camera” prompts, low light, fog, motion blur, CGI, digital art, scans, medical/scientific
  images, and aggressive smartphone computational photography.

The benchmark needs an explicit label policy for fully generated, local inpaint, generative
upscale, ordinary denoise, human digital art, CGI, and camera photographs of AI imagery.

## Browser matrix

Use a deterministic checksum scorer before involving the neural model. Known-color/quadrant
fixtures prove which pixels actually reach preprocessing.

- Bind every job to `{tabId, frameId, documentId, elementId, generation, currentSrc,
  intrinsicRect, renderedRect}` and discard out-of-order results.
- Test `<picture>`, `srcset`, DPR/zoom/media changes, lazy loading, SPA node recycling,
  same-URL byte changes, `blob:`/`data:` lifetime races, authenticated and signed URLs.
- Enumerate `<img>`, CSS image layers and pseudo-elements, canvas/WebGL, SVG, animated
  GIF/APNG/WebP, video posters/frames, open/closed shadow DOM, and cross-origin/about/blob frames.
- Test intrinsic versus displayed size, object-fit/object-position, clip paths, sprites,
  transforms, occlusion, multi-monitor scaling, nested scrolling, and visual viewport changes.
- Treat page badges as convenience UI. Store authoritative state in extension context and show
  it in trusted extension-owned UI.
- Bound bytes, decoded pixels, dimensions, queue depth, concurrency, retries, and decode time.
  Malformed/polyglot/decompression-bomb inputs must be isolated and labeled unsupported.
- Verify WebGPU fp16 and WASM-int8 paths, browser decoding/color/alpha/orientation, service-worker
  suspension, offscreen recreation, GPU device loss, navigation, offline restart, and OPFS quota.

## Adaptive and data attacks

- White-box and transfer attacks target the calibrated `p < 0.65` objective, not generic loss.
  Use expectation over exact browser resize/decode, codec, crop, and both runtime backends.
- Attack every ensemble member and its aggregation jointly. Architectural diversity alone is
  not adversarial robustness.
- Sweep 0.60–0.70 to expose threshold cliffs and test display rounding against internal decisions.
- Simulate source-local label flips, clean-label poisons, duplicated poison clusters, and visible
  and semantic triggers at 0.1/0.5/1/2% of training data.
- Train a metadata-only leakage probe from codec, dimensions, aspect ratio, file size, EXIF,
  color profile, alpha, and quantization estimates. Paired re-encodes should preserve decisions.
- Add embedding-based sibling/scene deduplication. Exact hashes and pHash alone do not catch
  crops, alternate exports, or semantically duplicated prompt/image families.
- Bootstrap hierarchically by generator/source and then original image. Image-only bootstrap
  does not represent uncertainty over unseen generators.

## Metrics

Every attack report includes:

- attack success among originally correct fakes;
- fixed-0.65 robust balanced accuracy with matched attacked reals;
- worst-variant-per-image and best-of-N evasion yield;
- fake recall, real specificity, score margin, distortion, query count, and latency;
- generator/source/resolution/AI-area/acquisition/runtime slices;
- hierarchical confidence intervals and paired deltas against the champion;
- exact-browser confirmation. PyTorch-only successes or defenses do not count.

## Initial release gates

1. Natural/web/hard BA lower 95% CI ≥0.75; clean champion regression ≤1 point.
2. Worst declared ordinary laundering BA ≥0.75; no critical generator recall below 0.60.
3. Zero stale bindings under 10 Hz source mutation.
4. Every eligible visible surface has an explicit terminal/pending state; no silent omissions.
5. Rendered fixture checksum matches the pixels entering preprocessing.
6. Mutation/load storms keep memory bounded and foreground latency within the declared SLO.
7. Damaged/partial assets fail closed; verified cold restart works with networking blocked.
8. Zero decision mismatches across WebGPU/WASM golden and adversarial parity corpora.
9. Metadata-only AUROC ≤0.60; metadata edits cause <2% flips; equivalent codec pairs <5%.
10. Do not claim adversarial robustness unless a diverse adaptive suite passes. Aspirational
    gates are robust BA ≥0.75 at 2/255 and ≥0.70 at 4/255.

## Execution order

1. Multi-hop laundering and codec/resize chains.
2. Partial composites and multi-scale/tiled aggregation.
3. Browser acquisition, mutation, and coverage fixtures.
4. Screenshot and physical recapture.
5. Generative restoration/img2img laundering and best-of-N selection.
6. Exact-graph adaptive attacks and poisoning/leakage audits.

## Implemented stack

- `configs/redteam.yaml` defines deterministic `smoke` and `core` attack profiles.
- `aiblink-val infer --attack-config ...` executes the registry through the actual model path.
- `aiblink-val redteam-report` freezes the chosen calibrator and reports per-case/family,
  any-attack evasion, worst-variant-per-image metrics, and hierarchical uncertainty.
- `aiblink-val metadata-audit` cross-validates a metadata-only leakage probe.
- `aiblink-val whitebox-eval --attack-config ...` executes exact-model PGD with each registry
  case's epsilon, step count, and restart count; the report rejects parameter mismatches.
- `browser_fixtures/` provides rendered sentinel fixtures and state/coverage contracts.

Partial real/fake composites at 5/10/25% synthetic area are now executable in the core profile.
True physical/platform recapture, generative laundering, best-of-N generation, poisoning
retraining, and exact extension-runtime adversarial confirmation still require captured external
cohorts, generation jobs, a training pipeline, or the eventual extension. Their absence is an
explicit failing completeness gate or documented pending suite, never silently ignored.

## References

- [Community Forensics robustness evaluation](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.pdf)
- [RRDataset real-world transmission and recapture benchmark](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.pdf)
- [Any-resolution spectral detector](https://openaccess.thecvf.com/content/CVPR2025/papers/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.pdf)
- [AutoAttack](https://proceedings.mlr.press/v119/croce20b.html) and [adaptive evaluation of obfuscated gradients](https://proceedings.mlr.press/v80/athalye18a.html)
- [Chrome content-script frames](https://developer.chrome.com/docs/extensions/reference/manifest/content-scripts), [network requests](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests), [offscreen documents](https://developer.chrome.com/docs/extensions/reference/api/offscreen), and [service workers](https://developer.chrome.com/docs/extensions/develop/migrate/to-service-workers)
