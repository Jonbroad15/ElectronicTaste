# MERT On-Device Spike: Findings

**Date**: 2026-05-21  
**Model**: MERT-v1-95M (94.4M params, HuggingFace `m-a-p/MERT-v1-95M`)  
**Host for conversion**: Apple Silicon Mac (macOS darwin), Python 3.10, torch 2.11, coremltools 9.0, onnxruntime 1.23.2

---

## Recommendation: **GO for iOS (on-device). CONDITIONAL for Android.**

Phase 4 cloud infrastructure **can be deferred** for iOS. MERT-v1-95M converts cleanly to CoreML, passes accuracy parity at both FP16 and INT8, and at 95 MB (INT8) fits comfortably within App Store limits. The on-device path also eliminates the connectivity problem — clubs and festivals have poor cell signal, which is exactly when users need the app.

Android is viable in principle (ORT FP32 works) but requires one additional step before it can be declared production-ready: the INT8 ORT model uses a `ConvInteger` op not yet supported by the ORT CPU provider on the desktop test host. This must be validated on a physical Android device against the ORT Android runtime before the cloud fallback can be dropped.

---

## Conversion Results

| Variant | Format | Size | Conversion |
|---|---|---|---|
| FP32 | CoreML `.mlpackage` | 377.8 MB | ✅ Success |
| **FP16** | **CoreML `.mlpackage`** | **189.0 MB** | **✅ Success** |
| **INT8** | **CoreML `.mlpackage`** | **95.3 MB** | **✅ Success** |
| FP32 | ONNX (+ .data sidecar) | 361.7 MB | ✅ Success |
| INT8 | ONNX (ORT quantized) | 94.9 MB | ✅ Success (see caveat) |

**CoreML conversion path**: `torch.export.export` → `run_decompositions({})` → `ct.convert()`.  
`torch.jit.trace` fails on MERT's dynamic attention masking (`create_bidirectional_mask`); `torch.export` via dynamo is the correct path for coremltools 8+.

**ONNX conversion path**: `torch.onnx.export` (dynamo backend, opset 18) → weights in external `.data` sidecar (onnx 1.21+ behaviour for large models). The INT8 ORT model was produced via `quant_pre_process` + `quantize_dynamic`; the FP32 must ship as two files (`mert.onnx` + `mert.onnx.data`), which ONNX Runtime supports natively.

---

## Accuracy Parity (30s clips, 3 samples, cosine similarity vs PyTorch baseline)

| Variant | Cosine Sim | Threshold | Pass? |
|---|---|---|---|
| ONNX FP32 | **1.0000** | ≥ 0.99 | ✅ PASS |
| ONNX INT8 | — | ≥ 0.97 | ⚠ Not tested (see caveat below) |
| CoreML FP16 | **0.9997** | ≥ 0.99 | ✅ PASS |
| CoreML INT8 | **0.9988** | ≥ 0.97 | ✅ PASS |

CoreML INT8 (0.9988) exceeds the FP16 threshold (0.99) — acceptable; use FP16 threshold for INT8 as a conservative bar.

---

## Latency Benchmarks (macOS Apple Silicon — proxy for on-device)

These times are measured on the conversion host (Apple Silicon Mac via CPU, no Neural Engine), **not** on a physical mobile device. They represent an upper bound — on-device Neural Engine execution will be substantially faster.

| Variant | Latency (30s clip) | Notes |
|---|---|---|
| PyTorch baseline | 898ms | MPS not used (CPU only) |
| ONNX FP32 | 2514ms | ORT CPU |
| CoreML FP16 | **1137ms** | macOS CoreML CPU path |
| CoreML INT8 | 2313ms | macOS CoreML CPU path |

CoreML FP16 is the fastest converted variant at 1.1s on the Mac CPU path. The Neural Engine on an iPhone 14+ executes transformer layers 3–5× faster than the CPU path, suggesting **real device latency for a 30s clip should be well under 10 seconds** (likely 200–500ms on flagship). On-device benchmarks on physical hardware are required to confirm this.

---

## Gotchas & Obstacles

1. **`torch.jit.trace` fails on MERT.** MERT's `create_bidirectional_mask` uses data-dependent control flow. Must use `torch.export.export(strict=False)` + `run_decompositions({})`. This is the blocking issue that would have stopped a naive conversion attempt.

2. **coremltools 7+ dropped ONNX source.** Cannot use `ct.convert(onnx_path, source="onnx")`. Must go directly from the PyTorch `ExportedProgram`.

3. **ONNX FP32 uses external data sidecar.** torch 2.11's dynamo exporter writes large models as a graph file + `.data` sidecar. Android deployment ships both files; `onnx.save(load_external_data=True, save_as_external_data=False)` silently drops weights for models over ~300 MB (protobuf byte limit). The correct FP32 Android path is to copy both sidecar files.

4. **ORT INT8 `ConvInteger` not supported on CPU provider on this host.** `onnxruntime.quantization.quantize_dynamic` produces `ConvInteger` ops that ORT 1.23.2 CPU does not implement on macOS. The model file is syntactically valid; this must be retested on a physical Android device using the ORT Android AAR (`onnxruntime-android`), which has a broader op set.

5. **CoreML fixed input shape.** The exported `.mlpackage` is fixed to the clip length used during export (30s = 720000 samples). If variable clip lengths are needed in the mobile app, the model must be re-exported with `ct.RangeDim`. For the MVP, a fixed 30s capture window is acceptable.

---

## Model Size Summary (App Bundle Impact)

| Recommended variant | Platform | Size | Status |
|---|---|---|---|
| CoreML INT8 | iOS | **95.3 MB** | ✅ Under 200 MB limit |
| ORT INT8 ONNX | Android | **94.9 MB** | ✅ Under 200 MB limit (pending device validation) |

---

## Outstanding Tasks Before Production

- [ ] Benchmark CoreML FP16/INT8 on physical iPhone (12 and 14+) via a minimal Swift test harness using Instruments.
- [ ] Validate ORT INT8 ONNX on physical Android device (Galaxy S23 + Pixel 6a) — specifically confirm `ConvInteger` is supported by the ORT Android AAR.
- [ ] Re-export CoreML with `ct.RangeDim` input if variable clip lengths are needed.
- [ ] Integrate CoreML model into Phase 5 mobile app (replace placeholder API call with on-device `MLModel.prediction()`).
- [ ] If Android ORT INT8 is confirmed working: remove Phase 4 cloud GPU provisioning from scope. Retain cloud infra only for model retraining / RLHF (Phase 6).
