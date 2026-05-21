#!/usr/bin/env python3
"""
Verify that ONNX and CoreML models produce embeddings within acceptable
cosine-similarity tolerance of the PyTorch baseline.

Thresholds (from spec):
  FP16: mean cosine similarity >= 0.99
  INT8: mean cosine similarity >= 0.97

Usage:
    python scripts/parity_check.py
    python scripts/parity_check.py --clip-seconds 5 --n-clips 10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

SAMPLE_RATE = 24000
MODELS_DIR  = Path("models")


class MERTEmbedder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        out = self.model(waveform, output_hidden_states=False)
        return out.last_hidden_state.mean(dim=1)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a * b, axis=-1)))


def run_pytorch(wrapper, clips):
    embeddings = []
    times = []
    with torch.no_grad():
        for clip in clips:
            t0 = time.perf_counter()
            emb = wrapper(clip).numpy()
            times.append((time.perf_counter() - t0) * 1000)
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0), np.mean(times)


def run_onnx(onnx_path: Path, clips):
    import onnxruntime as ort
    # For models with external .data sidecars, ORT resolves them relative
    # to the directory containing the .onnx file.
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    embeddings = []
    times = []
    for clip in clips:
        inp = {"waveform": clip.numpy()}
        t0 = time.perf_counter()
        out = sess.run(None, inp)
        times.append((time.perf_counter() - t0) * 1000)
        embeddings.append(out[0])
    return np.concatenate(embeddings, axis=0), np.mean(times)


def run_coreml(mlpackage_path: Path, clips):
    import coremltools as ct
    model = ct.models.MLModel(str(mlpackage_path))
    embeddings = []
    times = []
    for clip in clips:
        inp = {"waveform": clip.numpy()}
        t0 = time.perf_counter()
        out = model.predict(inp)
        times.append((time.perf_counter() - t0) * 1000)
        embeddings.append(list(out.values())[0])
    arr = np.array(embeddings)
    if arr.ndim == 3:
        arr = arr.squeeze(1)
    return arr, np.mean(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-seconds", type=int, default=5)
    parser.add_argument("--n-clips",      type=int, default=5)
    args = parser.parse_args()

    n_samples = args.clip_seconds * SAMPLE_RATE
    torch.manual_seed(42)
    clips = [torch.randn(1, n_samples) for _ in range(args.n_clips)]

    print(f"Loading PyTorch baseline ({args.n_clips} × {args.clip_seconds}s clips)...")
    base = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    base.eval()
    wrapper = MERTEmbedder(base)
    pt_embs, pt_ms = run_pytorch(wrapper, clips)
    print(f"  PyTorch mean latency: {pt_ms:.0f}ms")

    results = {}

    # ONNX FP32
    onnx_fp32 = MODELS_DIR / "mert.onnx"
    if onnx_fp32.exists():
        print(f"\nONNX FP32 ({onnx_fp32.stat().st_size / 1e6:.1f} MB)...")
        embs, ms = run_onnx(onnx_fp32, clips)
        sim = cosine_sim(pt_embs, embs)
        results["onnx_fp32"] = {"cosine_sim": round(sim, 4), "latency_ms": round(ms)}
        print(f"  cosine_sim={sim:.4f}  latency={ms:.0f}ms")

    # ONNX INT8
    onnx_int8 = MODELS_DIR / "android" / "mert_int8.onnx"
    if onnx_int8.exists():
        print(f"\nONNX INT8 ({onnx_int8.stat().st_size / 1e6:.1f} MB)...")
        try:
            embs, ms = run_onnx(onnx_int8, clips)
            sim = cosine_sim(pt_embs, embs)
            results["onnx_int8"] = {"cosine_sim": round(sim, 4), "latency_ms": round(ms)}
            flag = "" if sim >= 0.97 else "  ⚠ BELOW THRESHOLD (0.97)"
            print(f"  cosine_sim={sim:.4f}  latency={ms:.0f}ms{flag}")
        except Exception as e:
            results["onnx_int8"] = {"error": str(e)}
            print(f"  ERROR: {e}")

    # CoreML FP16
    coreml_fp16 = MODELS_DIR / "coreml" / "mert_fp16.mlpackage"
    if coreml_fp16.exists():
        print(f"\nCoreML FP16...")
        try:
            embs, ms = run_coreml(coreml_fp16, clips)
            sim = cosine_sim(pt_embs, embs)
            results["coreml_fp16"] = {"cosine_sim": round(sim, 4), "latency_ms": round(ms)}
            flag = "" if sim >= 0.99 else "  ⚠ BELOW THRESHOLD (0.99)"
            print(f"  cosine_sim={sim:.4f}  latency={ms:.0f}ms{flag}")
        except Exception as e:
            results["coreml_fp16"] = {"error": str(e)}
            print(f"  ERROR: {e}")

    # CoreML INT8
    coreml_int8 = MODELS_DIR / "coreml" / "mert_int8.mlpackage"
    if coreml_int8.exists():
        print(f"\nCoreML INT8...")
        try:
            embs, ms = run_coreml(coreml_int8, clips)
            sim = cosine_sim(pt_embs, embs)
            results["coreml_int8"] = {"cosine_sim": round(sim, 4), "latency_ms": round(ms)}
            flag = "" if sim >= 0.97 else "  ⚠ BELOW THRESHOLD (0.97)"
            print(f"  cosine_sim={sim:.4f}  latency={ms:.0f}ms{flag}")
        except Exception as e:
            results["coreml_int8"] = {"error": str(e)}
            print(f"  ERROR: {e}")

    print("\n=== Parity Summary ===")
    print(f"{'Variant':<18} {'Cosine Sim':>12} {'Latency (ms)':>14} {'Pass?':>8}")
    thresholds = {"onnx_fp32": 0.99, "onnx_int8": 0.97, "coreml_fp16": 0.99, "coreml_int8": 0.97}
    for variant, info in results.items():
        if "error" in info:
            print(f"  {variant:<16}  ERROR: {info['error'][:60]}")
            continue
        thresh = thresholds.get(variant, 0.97)
        passed = "PASS" if info["cosine_sim"] >= thresh else "FAIL"
        print(f"  {variant:<16}  {info['cosine_sim']:>12.4f}  {info['latency_ms']:>14}  {passed:>8}")

    out_path = MODELS_DIR / "parity_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"clip_seconds": args.clip_seconds, "n_clips": args.n_clips,
                   "pytorch_latency_ms": round(pt_ms), "results": results}, f, indent=2)
    print(f"\nResults saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
