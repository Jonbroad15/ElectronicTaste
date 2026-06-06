# MERT On-Device Spike: Validation & Success Criteria

To successfully complete this spike, the following conditions must be met:

1. **iOS conversion outcome documented**
   A CoreML `.mlpackage` has been produced, or every blocking operator incompatibility has been identified and a workaround has been attempted, with the result documented in `findings.md`.

2. **Android conversion outcome documented**
   A TFLite `.tflite` or ONNX Runtime `.onnx` model has been produced for Android, or every blocking conversion error has been identified and a fallback path attempted, with the result documented in `findings.md`.

3. **Quantized variants produced and sized for both platforms**
   FP16 and INT8 variants exist for iOS; INT8 variants exist for Android; all file sizes are recorded, and any INT8 accuracy delta outside the 0.97 threshold is documented as a known trade-off.

4. **On-device latency measured on physical iOS hardware**
   Inference time for a 30-second clip has been measured on at least one physical iOS device using Instruments; the result is recorded in `findings.md` alongside device model and iOS version.

5. **On-device latency measured on physical Android hardware**
   Inference time for a 30-second clip has been measured on at least one physical Android device using Android Profiler or ADB; the result is recorded in `findings.md` alongside device model, Android version, and runtime used (TFLite or ONNX Runtime).

6. **Memory usage confirmed within OS limits on both platforms**
   Peak RAM during inference is recorded for both iOS and Android; if it exceeds 1 GB on any tested device, this is flagged as a risk in the recommendation.

7. **Accuracy parity verified on both platforms**
   Cosine similarity between model embeddings and PyTorch baseline embeddings has been computed for the GTZAN test clips on both the CoreML and Android models, confirming functional equivalence at the chosen quantization level.

8. **Written Go / No-Go recommendation committed**
   `findings.md` is committed to this branch with a clear recommendation covering both platforms: (a) on-device is viable on both and Phase 4 cloud infra can be deferred/scoped down, (b) on-device is viable on one platform with a hybrid strategy proposed, or (c) cloud infra is required with specific blocking reasons stated for each platform.
