#!/usr/bin/env python3
"""Benchmark script — accuracy and speed checks against NFR thresholds.

Usage::

    # Full benchmark on held-out test split (recommended):
    python -m src.scripts.benchmark \\
        --data-dir data/gtzan_audio \\
        --split-manifest models/split_manifest.json \\
        --split test \\
        --classifier models/classifier_best.pt \\
        --label-encoder models/label_encoder.json \\
        --output results/benchmark.json

    # Full benchmark on entire directory (no split):
    python -m src.scripts.benchmark \\
        --data-dir data/gtzan_audio \\
        --classifier models/classifier_best.pt

    # Speed only:
    python -m src.scripts.benchmark \\
        --data-dir data/gtzan_audio \\
        --classifier models/classifier_best.pt \\
        --speed-only

Pass Criteria (from NFR-1 / NFR-2):
    • Top-1 accuracy          ≥ 70 %
    • Worst per-class accuracy ≥ 40 %
    • Mean inference time     ≤ 0.5 s / clip
    • Projected 200-file time ≤ 60 s
    • Peak memory usage       ≤ 4 GB

Exits with code 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from src.audio.preprocess import load_and_preprocess
from src.models.mert_encoder import MERTEncoder, get_device
from src.models.classifier import SubgenreClassifier
from src.training.dataset import LabelEncoder

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}

# ── NFR thresholds ────────────────────────────────────────────────────────────
NFR_MIN_ACCURACY       = 0.70   # overall top-1 accuracy
NFR_MIN_CLASS_ACCURACY = 0.40   # worst-class floor
NFR_MAX_INFERENCE_S    = 0.50   # seconds per 5-second clip
NFR_MAX_BATCH_200_S    = 60.0   # projected seconds for 200-file batch
NFR_MAX_MEMORY_GB      = 4.0    # peak memory during inference


def collect_samples(
    data_dir: Path,
    split_manifest: dict | None = None,
    split: str = "test",
) -> list[tuple[Path, str]]:
    """Collect labelled audio paths, optionally filtered to a split.

    Args:
        data_dir:       Root directory with per-class subdirectories.
        split_manifest: Loaded ``models/split_manifest.json`` dict, or None
                        to evaluate the entire directory.
        split:          Which split to use: ``"train"``, ``"val"``, or ``"test"``.
    """
    all_samples: list[tuple[Path, str]] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.glob("*")):
            if f.suffix.lower() in _AUDIO_EXTENSIONS:
                all_samples.append((f, class_dir.name))

    if split_manifest is None:
        return all_samples

    allowed = set(split_manifest[split])
    filtered = [(p, t) for p, t in all_samples if str(p) in allowed]
    return filtered


def run_inference(
    samples: list[tuple[Path, str]],
    encoder: MERTEncoder,
    classifier: SubgenreClassifier,
    label_enc: LabelEncoder,
    device: torch.device,
) -> tuple[list[dict], list[float], float]:
    """Run inference on all samples; return results, per-file timings, and peak RSS GB."""
    results: list[dict] = []
    timings: list[float] = []

    mem_before_gb = 0.0
    mem_peak_gb   = 0.0
    if _PSUTIL_AVAILABLE:
        proc = psutil.Process()
        mem_before_gb = proc.memory_info().rss / 1e9

    for i, (path, truth) in enumerate(samples, 1):
        t0 = time.time()
        waveform   = load_and_preprocess(str(path))
        embedding  = encoder.extract_embedding(waveform).to(device)
        predictions = classifier.predict_top3(embedding, label_enc.names)
        elapsed    = time.time() - t0

        timings.append(elapsed)
        results.append({"path": str(path), "truth": truth, "predictions": predictions})

        if _PSUTIL_AVAILABLE:
            mem_peak_gb = max(mem_peak_gb, proc.memory_info().rss / 1e9)

        if i % 20 == 0 or i == len(samples):
            print(f"  [{i:>4}/{len(samples)}] {path.name:<40} "
                  f"pred={predictions[0]['label']:<14} "
                  f"truth={truth:<14}  {elapsed:.2f}s")

    delta_gb = mem_peak_gb - mem_before_gb if _PSUTIL_AVAILABLE else 0.0
    return results, timings, max(delta_gb, 0.0)


def compute_metrics(
    results: list[dict],
    label_enc: LabelEncoder,
    timings: list[float],
    mem_delta_gb: float,
) -> dict:
    """Compute accuracy, per-class metrics, confusion, timing, and memory stats."""
    correct = 0
    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_total:   dict[str, int] = defaultdict(int)
    confusion: dict[str, int]         = defaultdict(int)

    for r in results:
        truth = r["truth"]
        top1  = r["predictions"][0]["label"]
        per_class_total[truth] += 1
        if top1 == truth:
            correct += 1
            per_class_correct[truth] += 1
        else:
            confusion[f"{truth} → {top1}"] += 1

    n = len(results)
    per_class_acc = {
        cls: {
            "correct":  per_class_correct[cls],
            "total":    per_class_total[cls],
            "accuracy": round(per_class_correct[cls] / per_class_total[cls], 4)
                        if per_class_total[cls] > 0 else 0.0,
        }
        for cls in label_enc.names
        if cls in per_class_total
    }

    return {
        "num_files":     n,
        "top1_accuracy": round(correct / n, 4) if n else 0.0,
        "per_class":     per_class_acc,
        "top_confusions": dict(sorted(confusion.items(), key=lambda x: -x[1])[:15]),
        "timing": {
            "mean_s":  round(statistics.mean(timings), 4),
            "stdev_s": round(statistics.stdev(timings) if len(timings) > 1 else 0.0, 4),
            "min_s":   round(min(timings), 4),
            "max_s":   round(max(timings), 4),
            "total_s": round(sum(timings), 2),
        },
        "memory": {
            "delta_gb":       round(mem_delta_gb, 3),
            "psutil_available": _PSUTIL_AVAILABLE,
        },
    }


def plot_confusion_matrix(metrics: dict, output_path: Path) -> None:
    """Save a confusion matrix PNG to ``output_path``."""
    if not _MATPLOTLIB_AVAILABLE:
        print("  (matplotlib not available — skipping confusion matrix plot)")
        return

    labels = sorted(metrics["per_class"].keys())
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    # Fill diagonal from per_class correct counts
    for i, lbl in enumerate(labels):
        if lbl in metrics["per_class"]:
            matrix[i][i] = metrics["per_class"][lbl]["correct"]

    # Fill off-diagonal from confusion dict
    for pair, count in metrics["top_confusions"].items():
        parts = pair.split(" → ")
        if len(parts) == 2 and parts[0] in labels and parts[1] in labels:
            matrix[labels.index(parts[0])][labels.index(parts[1])] = count

    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = matrix.max() * 0.5 if matrix.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                        color="white" if matrix[i][j] > thresh else "black", fontsize=8)

    plt.colorbar(im)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {output_path}")


def print_report(metrics: dict) -> None:
    print(f"\n{'═' * 62}")
    print("BENCHMARK REPORT")
    print(f"{'═' * 62}")
    print(f"  Files processed : {metrics['num_files']}")
    print(f"  Top-1 accuracy  : {metrics['top1_accuracy']:.1%}  "
          f"(threshold ≥ {NFR_MIN_ACCURACY:.0%})")

    print(f"\n  Per-class accuracy:")
    print(f"  {'Class':<16} {'Correct':>7} {'Total':>5} {'Acc':>6}")
    print(f"  {'─' * 36}")
    for cls, stats in sorted(metrics["per_class"].items()):
        flag = " ← BELOW FLOOR" if stats["accuracy"] < NFR_MIN_CLASS_ACCURACY else ""
        print(f"  {cls:<16} {stats['correct']:>7} {stats['total']:>5} "
              f"{stats['accuracy']:>5.1%}{flag}")

    print(f"\n  Top confusions:")
    for pair, count in list(metrics["top_confusions"].items())[:10]:
        print(f"    {pair}: {count}")

    t = metrics["timing"]
    projected_200 = t["mean_s"] * 200
    print(f"\n  Inference timing:")
    print(f"    Mean        : {t['mean_s']:.3f}s  (threshold ≤ {NFR_MAX_INFERENCE_S}s)")
    print(f"    Stdev       : {t['stdev_s']:.3f}s")
    print(f"    Min         : {t['min_s']:.3f}s")
    print(f"    Max         : {t['max_s']:.3f}s")
    print(f"    Total       : {t['total_s']:.1f}s  ({metrics['num_files']} files)")
    print(f"    Proj. 200   : {projected_200:.1f}s  (threshold ≤ {NFR_MAX_BATCH_200_S}s)")

    m = metrics["memory"]
    if m["psutil_available"]:
        print(f"\n  Peak memory Δ   : {m['delta_gb']:.2f} GB  "
              f"(threshold ≤ {NFR_MAX_MEMORY_GB} GB)")
    else:
        print(f"\n  Peak memory     : (install psutil to enable monitoring)")


def check_nfr(metrics: dict, speed_only: bool = False) -> list[str]:
    """Return a list of failing NFR checks (empty = all pass)."""
    failures: list[str] = []

    if not speed_only:
        acc = metrics["top1_accuracy"]
        if acc < NFR_MIN_ACCURACY:
            failures.append(f"Top-1 accuracy {acc:.1%} < {NFR_MIN_ACCURACY:.0%}")

        worst_cls, worst_acc = min(
            ((cls, s["accuracy"]) for cls, s in metrics["per_class"].items()),
            key=lambda x: x[1],
            default=("", 1.0),
        )
        if worst_acc < NFR_MIN_CLASS_ACCURACY:
            failures.append(
                f"Worst-class accuracy ({worst_cls}: {worst_acc:.1%}) "
                f"< {NFR_MIN_CLASS_ACCURACY:.0%}"
            )

    mean_t = metrics["timing"]["mean_s"]
    if mean_t > NFR_MAX_INFERENCE_S:
        failures.append(f"Mean inference time {mean_t:.3f}s > {NFR_MAX_INFERENCE_S}s")

    projected_200 = mean_t * 200
    if projected_200 > NFR_MAX_BATCH_200_S:
        failures.append(
            f"Projected 200-file batch {projected_200:.1f}s > {NFR_MAX_BATCH_200_S}s"
        )

    if metrics["memory"]["psutil_available"]:
        delta = metrics["memory"]["delta_gb"]
        if delta > NFR_MAX_MEMORY_GB:
            failures.append(f"Peak memory Δ {delta:.2f} GB > {NFR_MAX_MEMORY_GB} GB")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark accuracy and inference speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True,
                        help="Root directory with per-class audio subdirectories")
    parser.add_argument("--classifier", default="models/classifier_best.pt")
    parser.add_argument("--label-encoder", default="models/label_encoder.json")
    parser.add_argument("--split-manifest", default=None,
                        help="Path to models/split_manifest.json. "
                             "If provided, only the specified --split is evaluated.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate (requires --split-manifest)")
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--confusion-matrix", default="results/confusion_matrix.png",
                        help="Path to write confusion matrix PNG")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--speed-only", action="store_true",
                        help="Skip accuracy metrics; check speed NFRs only")
    args = parser.parse_args()

    device   = get_device(args.device)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Data dir   : {data_dir}")
    print(f"Classifier : {args.classifier}")

    split_manifest = None
    if args.split_manifest:
        split_manifest = json.loads(Path(args.split_manifest).read_text())
        print(f"Split      : {args.split} ({len(split_manifest[args.split])} files)\n")
    else:
        print(f"Split      : (all files — no manifest provided)\n")

    # ── Load models ───────────────────────────────────────────────
    encoder    = MERTEncoder(model_id=args.mert_model, device=device)
    classifier = SubgenreClassifier.load(args.classifier, device=device)
    label_enc  = LabelEncoder.load(args.label_encoder)

    # ── Collect samples ───────────────────────────────────────────
    samples = collect_samples(data_dir, split_manifest, args.split)
    print(f"Files found: {len(samples)}\n")
    if not samples:
        print("ERROR: no audio files found.")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────
    print("Running inference…")
    results, timings, mem_gb = run_inference(samples, encoder, classifier, label_enc, device)

    # ── Compute & display metrics ─────────────────────────────────
    metrics = compute_metrics(results, label_enc, timings, mem_gb)
    print_report(metrics)

    # ── Confusion matrix PNG ──────────────────────────────────────
    if not args.speed_only:
        plot_confusion_matrix(metrics, Path(args.confusion_matrix))

    # ── NFR checks ────────────────────────────────────────────────
    failures = check_nfr(metrics, speed_only=args.speed_only)

    print(f"\n{'═' * 62}")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  ✗  {f}")
        print(f"{'═' * 62}\n")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED ✓")
        print(f"{'═' * 62}\n")

    # ── Save JSON ─────────────────────────────────────────────────
    with open(output_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Benchmark results → {output_path}")


if __name__ == "__main__":
    main()
