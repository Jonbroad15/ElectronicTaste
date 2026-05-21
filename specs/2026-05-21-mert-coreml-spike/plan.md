# MERT CoreML Spike: Execution Plan

## Step 1: CoreML Conversion Attempt
- Install `coremltools` and `torch` in the spike environment.
- Load MERT-v1-95M from HuggingFace and trace the encoder with `torch.jit.trace` using a dummy 30-second waveform (24 kHz, 720 000 samples).
- Convert the traced model to CoreML via `coremltools.convert()`.
- Document every operator that fails to convert and research available workarounds (custom ops, model surgery, alternative export paths such as ONNX → CoreML via `onnx-coreml` or `coremltools` ONNX importer).
- **Output**: A CoreML `.mlpackage` file, or a detailed list of blocking conversion errors and proposed workarounds.

## Step 2: Quantization
- Produce an FP16 variant using `coremltools.optimize.coreml.linear_quantize_weights`.
- Produce an INT8 variant and measure accuracy delta against the FP32 baseline embeddings.
- Record file size for each variant.
- **Output**: Two quantized `.mlpackage` files with size and accuracy delta documented.

## Step 3: On-Device Benchmark (iOS)
- Build a minimal Swift/Xcode test harness that loads the `.mlpackage`, feeds it a 30-second WAV file, and reports wall-clock inference time and peak memory via `os_signpost` / Instruments.
- Run on at least two devices: one flagship (iPhone 14 or later) and one mid-range (iPhone 12 or 13).
- Record: inference latency, peak RAM, CPU vs Neural Engine utilisation.
- **Output**: Benchmark table with device, model variant, latency, and peak RAM.

## Step 4: Accuracy Parity Verification
- Extract embeddings from the CoreML model for the GTZAN test clips used in `scripts/mert_prototype.py`.
- Compare cosine similarity between CoreML embeddings and the original PyTorch embeddings.
- Acceptable threshold: mean cosine similarity ≥ 0.99 for FP16, ≥ 0.97 for INT8.
- **Output**: Parity report confirming which quantization level is safe to use.

## Step 5: Go / No-Go Decision
- Synthesise benchmark and parity results into a written recommendation.
- If viable: outline the integration path into the mobile app (Phase 5), estimate model download size impact on app bundle, and confirm Phase 4 cloud infra can be deferred or scoped down to training/retraining only.
- If not viable: document the specific blocking constraints (operator gaps, latency, memory) and confirm Phase 4 proceeds as planned.
- Update `specs/2026-05-21-cloud-infra/` if the decision changes its scope.
- **Output**: A `findings.md` in this spec directory with the recommendation and supporting data.
