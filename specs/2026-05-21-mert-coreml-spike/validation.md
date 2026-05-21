# MERT CoreML Spike: Validation & Success Criteria

To successfully complete this spike, the following conditions must be met:

1. **Conversion outcome documented**
   A CoreML `.mlpackage` has been produced, or every blocking operator incompatibility has been identified and a workaround has been attempted, with the result documented in `findings.md`.

2. **Quantized variants produced and sized**
   FP16 and INT8 `.mlpackage` files exist, their sizes are recorded, and the INT8 accuracy delta (cosine similarity vs FP32) is within the 0.97 threshold or the threshold deviation is documented as a known trade-off.

3. **On-device latency measured on physical hardware**
   Inference time for a 30-second clip has been measured on at least one physical iOS device using Instruments; the result is recorded in `findings.md` alongside device model and iOS version.

4. **Memory usage confirmed within iOS limits**
   Peak RAM during inference is recorded; if it exceeds 1 GB on any tested device, this is flagged as a risk in the recommendation.

5. **Accuracy parity verified against PyTorch baseline**
   Cosine similarity between CoreML and PyTorch embeddings has been computed for the GTZAN test clips, confirming the converted model is functionally equivalent at the chosen quantization level.

6. **Written Go / No-Go recommendation committed**
   `findings.md` is committed to this branch with a clear recommendation: either (a) on-device is viable and Phase 4 cloud infra can be deferred/scoped down, or (b) cloud infra is required with specific reasons stated.
