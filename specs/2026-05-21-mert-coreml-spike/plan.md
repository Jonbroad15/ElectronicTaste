# MERT On-Device Spike: Execution Plan

## Step 1: iOS CoreML Conversion
- Install `coremltools` and `torch` in the spike environment.
- Load MERT-v1-95M from HuggingFace and trace the encoder with `torch.jit.trace` using a dummy 30-second waveform (24 kHz, 720 000 samples).
- Convert the traced model to CoreML via `coremltools.convert()`.
- Document every operator that fails to convert and research available workarounds (custom ops, model surgery, alternative export paths such as ONNX → CoreML via `coremltools` ONNX importer).
- **Output**: A CoreML `.mlpackage` file, or a detailed list of blocking conversion errors and proposed workarounds.

## Step 2: Android Conversion (TFLite / ONNX Runtime)
- Export MERT-v1-95M to ONNX via `torch.onnx.export` using the same dummy 30-second waveform.
- Attempt conversion to TFLite via `onnx-tf` + TFLite converter; document any operator gaps.
- As a fallback, package the ONNX model for Android using ONNX Runtime for Android (`onnxruntime-android`), which avoids a TFLite conversion step.
- Document which path succeeds and any workarounds required.
- **Output**: A `.tflite` or `.onnx` model ready for Android deployment, or a detailed list of blocking errors.

## Step 3: Quantization (Both Platforms)
- iOS: produce FP16 and INT8 variants using `coremltools.optimize.coreml.linear_quantize_weights`.
- Android: produce INT8 variants via ONNX Runtime quantization (`onnxruntime.quantization`) or TFLite post-training quantization.
- Record file size for each variant on each platform.
- **Output**: Quantized model files for both platforms with sizes documented.

## Step 4: On-Device Benchmark — iOS
- Build a minimal Swift/Xcode test harness that loads the `.mlpackage`, feeds it a 30-second WAV file, and reports wall-clock inference time and peak memory via `os_signpost` / Instruments.
- Run on at least two devices: one flagship (iPhone 14 or later) and one mid-range (iPhone 12 or 13).
- Record: inference latency, peak RAM, CPU vs Neural Engine utilisation.
- **Output**: Benchmark table with device, model variant, latency, and peak RAM.

## Step 5: On-Device Benchmark — Android
- Build a minimal Android test harness (Kotlin or a Python ADB script) that loads the TFLite or ONNX Runtime model, feeds it a 30-second WAV file, and reports wall-clock inference time and peak memory via Android Profiler or `dumpsys meminfo`.
- Run on at least two devices: one flagship (Samsung Galaxy S23 or later) and one mid-range (Pixel 6a or equivalent).
- Record: inference latency, peak RAM, CPU vs NPU utilisation.
- **Output**: Benchmark table with device, model variant, latency, and peak RAM — comparable format to the iOS table.

## Step 6: Accuracy Parity Verification
- Extract embeddings from both the CoreML and Android models for the GTZAN test clips used in `scripts/mert_prototype.py`.
- Compare cosine similarity between each platform's embeddings and the original PyTorch embeddings.
- Acceptable threshold: mean cosine similarity ≥ 0.99 for FP16, ≥ 0.97 for INT8 on both platforms.
- **Output**: Parity report confirming which quantization level is safe to use on each platform.

## Step 7: Go / No-Go Decision
- Synthesise benchmark and parity results across both platforms into a written recommendation.
- If viable on both: outline the integration path into the mobile app (Phase 5), estimate model download size impact, and confirm Phase 4 cloud infra can be deferred or scoped down to training/retraining only.
- If viable on one platform only: document the discrepancy and propose a hybrid strategy (on-device for one platform, cloud fallback for the other).
- If not viable on either: document the specific blocking constraints and confirm Phase 4 proceeds as planned.
- Update `specs/2026-05-21-cloud-infra/` if the decision changes its scope.
- **Output**: A `findings.md` in this spec directory with the recommendation and supporting data for both platforms.
