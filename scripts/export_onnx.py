#!/usr/bin/env python3
"""
Export MERT-v1-95M to ONNX.

Wraps the encoder in a thin module that takes a raw waveform and returns
a mean-pooled 768-dim embedding, then traces and exports to ONNX with
dynamic sequence length so both 5s and 30s clips are valid inputs.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --clip-seconds 30 --out models/mert_30s.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

SAMPLE_RATE = 24000
DEFAULT_OUT = Path("models/mert.onnx")


class MERTEmbedder(nn.Module):
    """Thin wrapper: waveform → mean-pooled 768-dim embedding."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        out = self.model(waveform, output_hidden_states=False)
        # Mean-pool over time dimension: (B, T, 768) → (B, 768)
        return out.last_hidden_state.mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-seconds", type=int, default=30)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_samples = args.clip_seconds * SAMPLE_RATE

    print(f"Loading MERT-v1-95M...")
    base = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    base.eval()
    wrapper = MERTEmbedder(base)

    dummy = torch.randn(1, n_samples)
    print(f"Input shape: {dummy.shape}  ({args.clip_seconds}s at {SAMPLE_RATE}Hz)")

    # Warm-up + PyTorch baseline timing
    with torch.no_grad():
        t0 = time.perf_counter()
        baseline_emb = wrapper(dummy)
        torch_ms = (time.perf_counter() - t0) * 1000
    print(f"PyTorch inference: {torch_ms:.0f}ms  embedding shape: {baseline_emb.shape}")

    print(f"\nExporting to ONNX → {args.out} ...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(args.out),
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={
            "waveform":  {0: "batch", 1: "samples"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify ONNX model is well-formed
    import onnx
    model_proto = onnx.load(str(args.out))
    onnx.checker.check_model(model_proto)
    size_mb = args.out.stat().st_size / 1e6
    print(f"ONNX export OK — {size_mb:.1f} MB")

    return baseline_emb


if __name__ == "__main__":
    main()
