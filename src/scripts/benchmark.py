#!/usr/bin/env python3
"""Benchmark script — accuracy and speed checks against NFR thresholds.

Usage::

    # Full benchmark (accuracy + speed)
    python -m src.scripts.benchmark \\
        --data-dir data/gtzan_audio \\
        --classifier models/classifier_best.pt \\
        --label-encoder models/label_encoder.json \\
        --output results/benchmark.json

    # Speed only (no accuracy metrics)
    python -m src.scripts.benchmark \\
        --data-dir data/gtzan_audio \\
        --classifier models/classifier_best.pt \\
        --speed-only

Pass Criteria (from NFR-1 / NFR-2):
    • Top-1 accuracy          ≥ 70 %
    • Worst per-class accuracy ≥ 40 %
    • Mean inference time     ≤ 0.5 s / clip
    • 200-file batch time     ≤ 60 s

Exits with code 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from src.audio.preprocess import load_and_preprocess
from src.models.mert_encoder import MERTEncoder, get_device
from src.models.classifier import SubgenreClassifier
from src.training.dataset import LabelEncoder

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}

# ── NFR thresholds ────────────────────────────────────────────────────────────
NFR_MIN_ACCURACY       = 0.70   # overall top-1 accuracy
NFR_MIN_CLASS_ACCURACY = 0.40   # worst-class floor
NFR_MAX_INFERENCE_S    = 0.50   # seconds per 5-second clip
NFR_MAX_BATCH_200_S    = 60.0   # seconds for 200-file batch


def collect_samples(data_dir: Path) -> list[tuple[Path, str]]:
    """Collect all labelled audio paths from a structured directory."""
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.glob("*")):
            if f.suffix.lower() in _AUDIO_EXTENSIONS:
                samples.append((f, class_dir.name))
    return samples


def run_inference(
    samples: list[tuple[Path, str]],
    encoder: MERTEncoder,
    classifier: SubgenreClassifier,
    label_enc: LabelEncoder,
    device: torch.device,
) -> tuple[list[dict], list[float]]:
    """Run inference on all samples.

    Returns:
        results:  List of dicts with ``path``, ``truth``, ``predictions``.
        timings:  Per-file wall-clock inference times.
    """
    results: list[dict] = []
    timings: list[float] = []

    for i, (path, truth) in enumerate(samples, 1):
        t0 = time.time()
        waveform = load_and_preprocess(str(path))
        embedding = encoder.extract_embedding(waveform).to(device)
        predictions = classifier.predict_top3(embedding, label_enc.names)
        elapsed = time.time() - t0

        timings.append(elapsed)
        results.append({
            "path": str(path),
            "truth": truth,
            "predictions": predictions,
        })

        if i % 20 == 0 or i == len(samples):
            print(f"  [{i:>4}/{len(samples)}] {path.name:<40} "
                  f"pred={predictions[0]['label']:<14} "
                  f"truth={truth:<14}  {elapsed:.2f}s")

    return results, timings


def compute_metrics(
    results: list[dict],
    label_enc: LabelEncoder,
    timings: list[float],
) -> dict:
    """Compute accuracy, per-class metrics, confusion, and timing stats."""
    correct = 0
    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_total:   dict[str, int] = defaultdict(int)
    confusion: dict[str, int] = defaultdict(int)

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
            "correct": per_class_correct[cls],
            "total":   per_class_total[cls],
            "accuracy": round(per_class_correct[cls] / per_class_total[cls], 4)
            if per_class_total[cls] > 0 else 0.0,
        }
        for cls in label_enc.names
        if cls in per_class_total
    }

    import statistics
    return {
        "num_files": n,
        "top1_accuracy": round(correct / n, 4) if n else 0.0,
        "per_class": per_class_acc,
        "top_confusions": dict(
            sorted(confusion.items(), key=lambda x: -x[1])[:15]
        ),
        "timing": {
            "mean_s":   round(statistics.mean(timings), 4),
            "stdev_s":  round(statistics.stdev(timings) if len(timings) > 1 else 0.0, 4),
            "min_s":    round(min(timings), 4),
            "max_s":    round(max(timings), 4),
            "total_s":  round(sum(timings), 2),
        },
    }


def print_report(metrics: dict) -> None:
    print(f"\n{'═' * 60}")
    print("BENCHMARK REPORT")
    print(f"{'═' * 60}")
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
    print(f"\n  Inference timing:")
    print(f"    Mean   : {t['mean_s']:.3f}s  (threshold ≤ {NFR_MAX_INFERENCE_S}s)")
    print(f"    Stdev  : {t['stdev_s']:.3f}s")
    print(f"    Min    : {t['min_s']:.3f}s")
    print(f"    Max    : {t['max_s']:.3f}s")
    print(f"    Total  : {t['total_s']:.1f}s")


def check_nfr(metrics: dict) -> list[str]:
    """Return a list of failing NFR checks (empty = all pass)."""
    failures: list[str] = []

    acc = metrics["top1_accuracy"]
    if acc < NFR_MIN_ACCURACY:
        failures.append(
            f"Top-1 accuracy {acc:.1%} < {NFR_MIN_ACCURACY:.0%}"
        )

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
        failures.append(
            f"Mean inference time {mean_t:.3f}s > {NFR_MAX_INFERENCE_S}s"
        )

    if metrics["num_files"] >= 200:
        total_t = metrics["timing"]["total_s"]
        if total_t > NFR_MAX_BATCH_200_S:
            failures.append(
                f"200-file batch time {total_t:.1f}s > {NFR_MAX_BATCH_200_S}s"
            )

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
    parser.add_argument("--output", default="results/benchmark.json",
                        help="Path to write benchmark results JSON")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--speed-only", action="store_true",
                        help="Skip accuracy metrics; measure timing only")
    args = parser.parse_args()

    device = get_device(args.device)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Data dir   : {data_dir}")
    print(f"Classifier : {args.classifier}\n")

    # ── Load models ───────────────────────────────────────────────
    encoder = MERTEncoder(model_id=args.mert_model, device=device)
    classifier = SubgenreClassifier.load(args.classifier, device=device)
    label_enc = LabelEncoder.load(args.label_encoder)

    # ── Collect samples ───────────────────────────────────────────
    samples = collect_samples(data_dir)
    print(f"Files found: {len(samples)}\n")
    if not samples:
        print("ERROR: no audio files found.")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────
    print("Running inference…")
    results, timings = run_inference(samples, encoder, classifier, label_enc, device)

    # ── Compute metrics ───────────────────────────────────────────
    metrics = compute_metrics(results, label_enc, timings)
    print_report(metrics)

    # ── NFR checks ────────────────────────────────────────────────
    if args.speed_only:
        failures: list[str] = []
        mean_t = metrics["timing"]["mean_s"]
        if mean_t > NFR_MAX_INFERENCE_S:
            failures.append(f"Mean inference {mean_t:.3f}s > {NFR_MAX_INFERENCE_S}s")
    else:
        failures = check_nfr(metrics)

    print(f"\n{'═' * 60}")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  ✗  {f}")
        print(f"{'═' * 60}\n")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED ✓")
        print(f"{'═' * 60}\n")

    # ── Save JSON ─────────────────────────────────────────────────
    with open(output_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Benchmark results → {output_path}")


if __name__ == "__main__":
    main()
