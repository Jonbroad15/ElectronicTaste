#!/usr/bin/env python3
"""
Convert MERT-v1-95M to CoreML (.mlpackage) for iOS deployment.

coremltools 7+ dropped ONNX source support, so we trace the PyTorch
model directly via torch.jit.trace and pass it to ct.convert().

Produces FP32, FP16, and INT8 variants and reports file sizes.

Usage:
    python scripts/export_coreml.py
    python scripts/export_coreml.py --clip-seconds 30 --out-dir models/coreml/
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel

import coremltools as ct
import coremltools.optimize.coreml as cto

SAMPLE_RATE = 24000
DEFAULT_OUT  = Path("models/coreml")


class MERTEmbedder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        out = self.model(waveform, output_hidden_states=False)
        return out.last_hidden_state.mean(dim=1)


def size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6


def convert(clip_seconds: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = clip_seconds * SAMPLE_RATE

    print("Loading MERT-v1-95M for tracing...")
    base = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    base.eval()
    wrapper = MERTEmbedder(base)

    dummy = torch.randn(1, n_samples)
    print(f"Exporting with torch.export.export (input shape {dummy.shape})...")
    # torch.jit.trace fails on MERT's dynamic attention masking; torch.export
    # handles it via dynamo and is supported by coremltools 8+.
    exported = torch.export.export(wrapper, (dummy,), strict=False)
    # coremltools requires ATEN dialect; decompose TRAINING ops first.
    exported = exported.run_decompositions({})

    results = {}

    fp32_path = out_dir / "mert_fp32.mlpackage"
    if fp32_path.exists():
        print(f"  FP32 already exists, loading from disk...")
        mlmodel = ct.models.MLModel(str(fp32_path))
    else:
        print("Converting exported program → CoreML (FP32)...")
        mlmodel = ct.convert(
            exported,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
            inputs=[ct.TensorType(name="waveform", shape=dummy.shape)],
        )
        mlmodel.save(str(fp32_path))
    results["fp32"] = {"path": fp32_path, "size_mb": size_mb(fp32_path)}
    print(f"  FP32 → {fp32_path}  ({results['fp32']['size_mb']:.1f} MB)")

    # FP16: re-convert with float16 compute precision (coremltools applies
    # FP16 at conversion time, not as post-hoc weight quantization)
    print("Converting → CoreML FP16...")
    mlmodel_fp16 = ct.convert(
        exported,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="waveform", shape=dummy.shape)],
    )
    fp16_path = out_dir / "mert_fp16.mlpackage"
    mlmodel_fp16.save(str(fp16_path))
    results["fp16"] = {"path": fp16_path, "size_mb": size_mb(fp16_path)}
    print(f"  FP16 saved → {fp16_path}  ({results['fp16']['size_mb']:.1f} MB)")

    # INT8: weight-only post-training quantization on the FP32 model
    print("Quantizing → INT8...")
    op_config_int8 = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config_int8    = cto.OptimizationConfig(global_config=op_config_int8)
    mlmodel_int8   = cto.linear_quantize_weights(mlmodel, config=config_int8)
    int8_path = out_dir / "mert_int8.mlpackage"
    mlmodel_int8.save(str(int8_path))
    results["int8"] = {"path": int8_path, "size_mb": size_mb(int8_path)}
    print(f"  INT8 saved → {int8_path}  ({results['int8']['size_mb']:.1f} MB)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-seconds", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = convert(args.clip_seconds, args.out_dir)

    print("\n=== CoreML Model Sizes ===")
    for variant, info in results.items():
        print(f"  {variant.upper():5s}  {info['size_mb']:6.1f} MB  →  {info['path']}")

    return results


if __name__ == "__main__":
    main()
