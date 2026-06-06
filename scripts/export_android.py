#!/usr/bin/env python3
"""
Prepare MERT ONNX model for Android deployment.

Two paths are attempted in order:
  1. ONNX Runtime for Android — package the ONNX model directly; no
     conversion needed, ONNX Runtime handles execution on-device.
  2. TFLite via onnx-tf — convert ONNX → TF SavedModel → TFLite.
     Only attempted if onnx-tf is available.

INT8 quantization is applied to the ONNX model via onnxruntime.quantization.

Usage:
    python scripts/export_android.py
    python scripts/export_android.py --onnx models/mert.onnx --out-dir models/android/
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_ONNX = Path("models/mert.onnx")
DEFAULT_OUT  = Path("models/android")


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def quantize_onnx_int8(src: Path, dst: Path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.onnx_model import ONNXModel
    # The dynamo ONNX exporter leaves some weights as computed nodes rather
    # than initializers. Run ORT's optimizer first to fold constants, then
    # quantize from the optimised single-file model.
    import onnx
    from onnxruntime.quantization import quant_pre_process
    inline_path = dst.parent / "_inline_tmp.onnx"
    opt_path    = dst.parent / "_opt_tmp.onnx"
    try:
        model = onnx.load(str(src), load_external_data=True)
        onnx.save(model, str(inline_path), save_as_external_data=False)
        quant_pre_process(str(inline_path), str(opt_path), skip_symbolic_shape=True)
        quantize_dynamic(str(opt_path), str(dst), weight_type=QuantType.QInt8)
    finally:
        inline_path.unlink(missing_ok=True)
        opt_path.unlink(missing_ok=True)


def attempt_tflite(onnx_path: Path, out_dir: Path) -> dict | None:
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as e:
        print(f"  TFLite path skipped: {e}")
        return None

    print("  Converting ONNX → TF SavedModel...")
    tf_dir = out_dir / "mert_tf_savedmodel"
    tf_rep = prepare(onnx.load(str(onnx_path)))
    tf_rep.export_graph(str(tf_dir))

    print("  Converting TF SavedModel → TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = out_dir / "mert_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    return {"path": tflite_path, "size_mb": size_mb(tflite_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",    type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"ERROR: {args.onnx} not found — run export_onnx.py first.")
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Path 1: ONNX Runtime for Android (FP32).
    # ONNX Runtime natively supports external-data models; copy both the
    # graph file and the .data sidecar into the android output directory.
    ort_fp32      = args.out_dir / "mert_fp32.onnx"
    ort_fp32_data = args.out_dir / "mert_fp32.onnx.data"
    src_data = Path(str(args.onnx) + ".data")
    if not ort_fp32.exists():
        shutil.copy(args.onnx, ort_fp32)
        if src_data.exists():
            shutil.copy(src_data, ort_fp32_data)
    total_mb = size_mb(ort_fp32) + (size_mb(ort_fp32_data) if ort_fp32_data.exists() else 0)
    results["ort_fp32"] = {"path": ort_fp32, "size_mb": total_mb, "runtime": "ONNX Runtime"}
    print(f"ORT FP32 ready → {ort_fp32} (+.data)  ({total_mb:.1f} MB total)")

    # Path 1b: INT8 quantized ONNX for ONNX Runtime
    print("Quantizing ONNX → INT8 for ONNX Runtime...")
    ort_int8 = args.out_dir / "mert_int8.onnx"
    try:
        quantize_onnx_int8(args.onnx, ort_int8)
        results["ort_int8"] = {"path": ort_int8, "size_mb": size_mb(ort_int8), "runtime": "ONNX Runtime"}
        print(f"ORT INT8 ready → {ort_int8}  ({results['ort_int8']['size_mb']:.1f} MB)")
    except Exception as e:
        print(f"  INT8 quantization failed: {e}")

    # Path 2: TFLite (optional, requires onnx-tf + tensorflow)
    print("\nAttempting TFLite conversion (requires onnx-tf + tensorflow)...")
    tflite_result = attempt_tflite(args.onnx, args.out_dir)
    if tflite_result:
        results["tflite_int8"] = {**tflite_result, "runtime": "TFLite"}
        print(f"TFLite INT8 ready → {tflite_result['path']}  ({tflite_result['size_mb']:.1f} MB)")

    print("\n=== Android Model Sizes ===")
    for variant, info in results.items():
        print(f"  {variant:15s}  {info['size_mb']:6.1f} MB  runtime={info['runtime']}  →  {info['path']}")

    return results


if __name__ == "__main__":
    main()
