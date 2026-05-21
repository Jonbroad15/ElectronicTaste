# MERT CoreML Spike: Requirements

## Objectives
Determine whether MERT-v1-95M can run on-device on iOS before committing to cloud GPU infrastructure. The core question is: can a 30-second audio clip be classified into an electronic music subgenre entirely on a user's phone, within an acceptable latency, without an internet connection? The answer directly determines whether Phase 4 (cloud infra) is necessary or can be deferred.

## Specific Requirements
1. **Model Export**: Export MERT-v1-95M from the HuggingFace `transformers` format to Apple CoreML (`.mlpackage`), resolving any operator compatibility issues encountered during conversion.
2. **Quantization**: Produce at least one quantized variant (INT8 or FP16) to minimise on-device memory footprint and compare inference speed against the FP32 baseline.
3. **Inference Benchmark**: Measure end-to-end inference time for a 30-second audio clip on at least one physical iOS device (ideally a mid-range device, e.g. iPhone 12 or later, in addition to a current flagship).
4. **Memory Profiling**: Record peak RAM usage during inference to confirm the model fits within iOS memory limits without triggering jitter or termination.
5. **Accuracy Parity Check**: Verify that the CoreML model produces embeddings within acceptable numerical tolerance of the original PyTorch model on a small set of test clips (the GTZAN clips already used in `scripts/mert_prototype.py`).
6. **Go / No-Go Recommendation**: Produce a written recommendation on whether on-device inference is viable as the primary deployment path, or whether cloud infrastructure is required.

## Non-Functional Requirements
- The CoreML model binary (quantized) must be under 200 MB to remain a realistic App Store download.
- Inference on a 30-second clip must complete within 10 seconds on a flagship device to be considered viable for UX (users in a club will not wait longer).
- The spike must not introduce any new production dependencies — it is an investigation only.
