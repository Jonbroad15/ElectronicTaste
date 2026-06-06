"""
LLM Reasoning Prototype for EDM Subgenre Classification
========================================================
Phase 2, Step 3: Test whether Claude can classify electronic music
subgenres from structured audio feature descriptions, and compare
against a traditional Random Forest classifier.

Usage:
  # Synthetic data, rule-based sim:
  python src/llm_prototype.py

  # Real Beatport dataset, rule-based sim:
  python src/llm_prototype.py --dataset data/beatsdataset_full.csv

  # Real dataset + actual Claude LLM (samples 3 tracks/subgenre):
  python src/llm_prototype.py --dataset data/beatsdataset_full.csv --use-claude

  # Real dataset + Claude, 5 tracks per subgenre:
  python src/llm_prototype.py --dataset data/beatsdataset_full.csv --use-claude --llm-sample 5
"""

import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.semantic_translator import translate_features_to_description

warnings.filterwarnings('ignore')

# ── Electronic subgenre definitions with characteristic feature ranges ──
# These ranges are derived from music production knowledge and MIR research.
SUBGENRE_PROFILES = {
    "Deep House":       {"bpm": (118, 125), "centroid": (1500, 2500), "zcr": (0.03, 0.07), "energy": (0.03, 0.06)},
    "Tech House":       {"bpm": (124, 130), "centroid": (2000, 3500), "zcr": (0.05, 0.10), "energy": (0.05, 0.08)},
    "Techno":           {"bpm": (128, 140), "centroid": (2500, 4000), "zcr": (0.06, 0.12), "energy": (0.06, 0.10)},
    "Trance":           {"bpm": (130, 145), "centroid": (2000, 3500), "zcr": (0.04, 0.08), "energy": (0.05, 0.09)},
    "Drum and Bass":    {"bpm": (160, 180), "centroid": (3000, 5000), "zcr": (0.08, 0.15), "energy": (0.06, 0.10)},
    "Dubstep":          {"bpm": (138, 142), "centroid": (1000, 2500), "zcr": (0.05, 0.12), "energy": (0.07, 0.12)},
    "House":            {"bpm": (120, 130), "centroid": (2000, 3000), "zcr": (0.04, 0.08), "energy": (0.04, 0.07)},
    "Hardstyle":        {"bpm": (148, 155), "centroid": (2500, 4500), "zcr": (0.08, 0.14), "energy": (0.08, 0.12)},
    "Ambient":          {"bpm": (60, 100),  "centroid": (800, 2000),  "zcr": (0.01, 0.04), "energy": (0.005, 0.03)},
    "Progressive House":{"bpm": (124, 132), "centroid": (2000, 3500), "zcr": (0.04, 0.08), "energy": (0.04, 0.08)},
    "Electro House":    {"bpm": (126, 132), "centroid": (3000, 5000), "zcr": (0.07, 0.13), "energy": (0.07, 0.10)},
    "Minimal / Deep Tech":{"bpm": (122, 130), "centroid": (1500, 2800), "zcr": (0.03, 0.07), "energy": (0.03, 0.06)},
}


def generate_synthetic_dataset(n_per_genre=50, seed=42):
    """
    Generates a synthetic dataset mimicking the Beatport Top 100 structure.
    Each row has audio features + a ground truth subgenre label.
    """
    np.random.seed(seed)
    rows = []
    
    for genre, profile in SUBGENRE_PROFILES.items():
        for _ in range(n_per_genre):
            bpm = np.random.uniform(*profile["bpm"])
            centroid = np.random.uniform(*profile["centroid"])
            rolloff = centroid * np.random.uniform(1.3, 1.8)
            zcr = np.random.uniform(*profile["zcr"])
            energy = np.random.uniform(*profile["energy"])
            mfcc_std = np.random.uniform(8, 25)
            
            rows.append({
                "bpm": round(bpm, 2),
                "spectral_centroid_hz": round(centroid, 2),
                "spectral_rolloff_hz": round(rolloff, 2),
                "zero_crossing_rate": round(zcr, 4),
                "energy": round(energy, 4),
                "mfcc_std": round(mfcc_std, 2),
                "subgenre": genre,
            })
    
    df = pd.DataFrame(rows)
    return df


def load_beatport_dataset(csv_path):
    """
    Loads the real Beatport Kaggle dataset CSV (beatsdataset_full.csv).
    Maps its pyAudioAnalysis column names to our standardised feature names.

    Known column schema (from inspection):
      - '1-ZCRm'              -> zero_crossing_rate
      - '2-Energym'           -> energy
      - '4-SpectralCentroidm' -> spectral_centroid_hz  (normalised 0-1, scaled)
      - '5-SpectralSpreadm'   -> spectral_rolloff_hz
      - '69-BPM'              -> bpm
      - 'class'               -> subgenre
      - 'id'                  -> filename (dropped)
    """
    df = pd.read_csv(csv_path, header=0)
    
    # Explicit column mapping for this specific dataset
    column_mapping = {
        'class':               'subgenre',
        '69-BPM':              'bpm',
        '1-ZCRm':              'zero_crossing_rate',
        '2-Energym':           'energy',
        '4-SpectralCentroidm': 'spectral_centroid_hz',
        '5-SpectralSpreadm':   'spectral_rolloff_hz',
    }
    df = df.rename(columns=column_mapping)
    
    # Drop non-feature columns
    drop_cols = [c for c in ['Unnamed: 0', 'id'] if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Keep all numeric columns for the ML baseline (uses the full 92-feature set)
    df = df.dropna(subset=['subgenre'])
    df['bpm'] = pd.to_numeric(df['bpm'], errors='coerce')
    df = df.dropna(subset=['bpm'])

    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Subgenres ({df['subgenre'].nunique()}):")
    print(df['subgenre'].value_counts().to_string())
    
    return df


def rule_based_llm_classifier(description):
    """
    Rule-based baseline that simulates LLM reasoning.
    Kept for comparison — see claude_llm_classifier() for the real thing.
    """
    desc_lower = description.lower()
    if "extremely fast" in desc_lower:
        return "Drum and Bass"
    if "very fast" in desc_lower:
        if "drum and bass" in desc_lower or "jungle" in desc_lower:
            return "Drum and Bass"
        return "Hardstyle"
    if "fast tempo" in desc_lower:
        if "psytrance" in desc_lower:
            return "Trance"
        return "Hardstyle"
    if "very slow" in desc_lower:
        return "Ambient"
    if "slow-to-moderate" in desc_lower:
        if "dark" in desc_lower or "bass-heavy" in desc_lower:
            return "Ambient"
        return "Deep House"
    if "elevated tempo" in desc_lower:
        if "bright" in desc_lower or "harsh" in desc_lower:
            return "Techno"
        if "percussive" in desc_lower:
            return "Tech House"
        return "Trance"
    if "standard dance tempo" in desc_lower:
        if "bright" in desc_lower and "percussive" in desc_lower:
            return "Electro House"
        if "dark" in desc_lower or "bass-heavy" in desc_lower:
            return "Dubstep"
        if "warm" in desc_lower:
            if "tonal" in desc_lower and "smooth" in desc_lower:
                return "Deep House"
            return "Progressive House"
        return "Trance"
    if "moderate tempo" in desc_lower:
        if "dark" in desc_lower or "bass-heavy" in desc_lower:
            return "Minimal / Deep Tech"
        if "warm" in desc_lower:
            if "smooth" in desc_lower or "tonal" in desc_lower:
                return "Deep House"
            return "House"
        if "bright" in desc_lower:
            return "Tech House"
        return "House"
    return "House"


def claude_llm_classifier(prompt):
    """
    Calls the Claude CLI (`claude -p`) with the classification prompt and
    returns the stripped single-line prediction.
    Falls back to 'Unknown' if the call fails.
    """
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Strip ANSI escape codes and whitespace, take first non-empty line
        raw = result.stdout
        # Remove common ANSI sequences
        import re
        clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw)
        lines = [l.strip() for l in clean.splitlines() if l.strip()]
        return lines[0] if lines else "Unknown"
    except Exception as e:
        print(f"  ⚠️  Claude call failed: {e}")
        return "Unknown"


# Maps our human-readable classifier output to the dataset's CamelCase labels.
# Add entries here whenever the dataset introduces new label spellings.
LABEL_NORMALISE = {
    "Drum and Bass":      "DrumAndBass",
    "Deep House":         "DeepHouse",
    "Tech House":         "TechHouse",
    "Electro House":      "ElectroHouse",
    "Progressive House":  "ProgressiveHouse",
    "Minimal / Deep Tech":"Minimal",
    "Ambient":            "ElectronicaDowntempo",  # best proxy in this dataset
}


def normalise_prediction(label, dataset_labels):
    """
    Convert our classifier's output label to the closest label present in the
    actual dataset. Falls back to the original label if no mapping exists.
    """
    normalised = LABEL_NORMALISE.get(label, label)
    # If still not in the dataset, return as-is (will count as wrong, correctly)
    if normalised in dataset_labels:
        return normalised
    return label


def build_llm_prompt(description, subgenre_list):
    """
    Constructs the classification prompt for Claude (or Qwen in production).
    The subgenre list is passed in so it always matches the actual dataset labels.
    """
    genres_str = ", ".join(sorted(subgenre_list))
    
    prompt = (
        "You are an expert electronic music classifier. "
        "Given the following audio analysis of a track, predict which electronic "
        "music subgenre it belongs to.\n\n"
        "## Audio Analysis\n"
        f"{description}\n\n"
        "## Possible Subgenres\n"
        f"{genres_str}\n\n"
        "## Instructions\n"
        "1. Consider the tempo (BPM) as the primary differentiator between subgenres.\n"
        "2. Use spectral brightness and energy to distinguish subgenres with similar BPMs.\n"
        "3. Consider the percussive vs tonal balance.\n"
        "4. Your response MUST be EXACTLY one of the subgenre names listed above — "
        "copy it verbatim, no extra words, no punctuation.\n\n"
        "Subgenre:"
    )
    return prompt


def run_prototype(dataset_path=None, use_claude=False, llm_sample=3):
    """
    Main prototype execution.

    Args:
        dataset_path: Path to a Beatport CSV. None = synthetic data.
        use_claude:   If True, call Claude CLI for LLM classification.
                      If False, use the rule-based simulator.
        llm_sample:   Number of tracks per subgenre to send to Claude
                      (to keep runtime/cost reasonable). Default: 3.
    """
    print("=" * 70)
    print("  Electric Taste — LLM Reasoning Model Prototype")
    print("  Phase 2, Step 3: Subgenre Classification via Feature Descriptions")
    print("=" * 70)
    
    # ── Step 1: Load or generate dataset ──
    if dataset_path and os.path.exists(dataset_path):
        print(f"\n📂 Loading real dataset from: {dataset_path}")
        df = load_beatport_dataset(dataset_path)
    else:
        print("\n🎵 Generating synthetic Beatport-style dataset (50 tracks per subgenre)...")
        df = generate_synthetic_dataset(n_per_genre=50)
        print(f"   Generated {len(df)} tracks across {df['subgenre'].nunique()} subgenres")
    
    # ── Step 2: Identify feature columns ──
    # LLM pipeline uses the 6 semantically meaningful features
    llm_feature_cols = ['bpm', 'spectral_centroid_hz', 'spectral_rolloff_hz',
                        'zero_crossing_rate', 'energy', 'mfcc_std']
    llm_features = [c for c in llm_feature_cols if c in df.columns]

    # ML baseline uses ALL numeric columns for maximum accuracy
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    ml_features = [c for c in all_numeric if c != 'subgenre']

    print(f"\n📐 LLM pipeline features:  {len(llm_features)} cols  {llm_features}")
    print(f"📐 ML baseline features:   {len(ml_features)} cols (full feature set)")

    # ── Step 3: Split into train/test ──
    y = df['subgenre']
    X_llm = df[llm_features]
    X_ml  = df[ml_features]

    _, X_llm_test, _, y_test = train_test_split(
        X_llm, y, test_size=0.3, random_state=42, stratify=y
    )
    X_ml_train, X_ml_test, y_ml_train, y_ml_test = train_test_split(
        X_ml, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 Dataset split: {len(X_ml_train)} train, {len(X_ml_test)} test")

    
    # ── Step 3: Semantic Translation + LLM Classification ──
    dataset_labels = sorted(y.unique().tolist())
    llm_label = "Claude (claude -p)" if use_claude else "Rule-Based Simulator"

    print("\n" + "─" * 70)
    print(f"  TEST A: Semantic Translation → LLM Reasoning ({llm_label})")
    print("─" * 70)

    # When using Claude, sample a fixed number of tracks per subgenre
    # to keep runtime reasonable (full 690-track run would take ~15 min).
    if use_claude:
        test_df = pd.concat([
            X_llm_test.join(y_test)
              .groupby('subgenre', group_keys=False)
              .apply(lambda g: g.sample(min(llm_sample, len(g)), random_state=42))
        ])
        X_llm_eval = test_df.drop(columns=['subgenre'])
        y_llm_eval = test_df['subgenre']
        n_calls = len(X_llm_eval)
        print(f"  Using Claude CLI — sampling {llm_sample} tracks × {len(dataset_labels)} subgenres = {n_calls} API calls")
    else:
        X_llm_eval = X_llm_test
        y_llm_eval = y_test

    llm_predictions = []
    sample_prompts  = []

    for i, (idx, row) in enumerate(X_llm_eval.iterrows()):
        features    = row.to_dict()
        description = translate_features_to_description(features)
        prompt      = build_llm_prompt(description, dataset_labels)

        if use_claude:
            raw = claude_llm_classifier(prompt)
            # Claude should return one of the dataset labels verbatim
            # but strip stray whitespace / punctuation just in case
            raw = raw.strip().rstrip('.')
            prediction = raw if raw in dataset_labels else normalise_prediction(raw, set(dataset_labels))
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{n_calls}] actual={y_llm_eval.iloc[i]:25s}  predicted={prediction}")
        else:
            raw        = rule_based_llm_classifier(description)
            prediction = normalise_prediction(raw, set(dataset_labels))

        llm_predictions.append(prediction)

        if len(sample_prompts) < 3:
            sample_prompts.append({
                "actual":      y_llm_eval.iloc[len(sample_prompts)],
                "predicted":   prediction,
                "description": description,
                "prompt":      prompt,
            })

    llm_accuracy = accuracy_score(y_llm_eval, llm_predictions)
    print(f"\n✅ {llm_label} Accuracy: {llm_accuracy:.1%}  (on {len(y_llm_eval)} tracks)")
    print(f"\nClassification Report:")
    print(classification_report(y_llm_eval, llm_predictions, zero_division=0))
    
    # ── Step 4: Traditional ML Baseline (Random Forest, full 92-feature set) ──
    print("─" * 70)
    print("  TEST B: Traditional ML Baseline (Random Forest — full 92 features)")
    print("─" * 70)
    
    scaler = StandardScaler()
    X_ml_train_scaled = scaler.fit_transform(X_ml_train)
    X_ml_test_scaled  = scaler.transform(X_ml_test)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15)
    rf.fit(X_ml_train_scaled, y_ml_train)
    rf_predictions = rf.predict(X_ml_test_scaled)
    
    rf_accuracy = accuracy_score(y_ml_test, rf_predictions)
    print(f"\n✅ Random Forest Accuracy: {rf_accuracy:.1%}")
    print(f"\nClassification Report:")
    print(classification_report(y_ml_test, rf_predictions, zero_division=0))
    
    # ── Step 5: Feature Importance (top 10 from RF) ──
    print("─" * 70)
    print("  Top 10 Feature Importances (Random Forest)")
    print("─" * 70)
    importances = sorted(
        zip(ml_features, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:10]
    for feat, imp in importances:
        bar = "█" * int(imp * 100)
        print(f"  {feat:40s} {imp:.4f} {bar}")
    
    # ── Step 6: Save sample prompts for review ──
    print("\n" + "─" * 70)
    print("  Sample LLM Prompts (for review / future API integration)")
    print("─" * 70)
    
    for i, sp in enumerate(sample_prompts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Actual: {sp['actual']}")
        print(f"Predicted: {sp['predicted']}")
        print(f"Description: {sp['description']}")
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  LLM Reasoning (rule-based sim):  {llm_accuracy:.1%}")
    print(f"  Random Forest Baseline:          {rf_accuracy:.1%}")
    print()
    
    if llm_accuracy > 0.50:
        print("  ✅ The semantic translation → LLM reasoning approach is VIABLE.")
        print("     The rule-based simulator achieves meaningful accuracy, and a real")
        print("     LLM (e.g., Qwen) with proper fine-tuning should do significantly better.")
    else:
        print("  ⚠️  The rule-based LLM simulator under-performs vs. Random Forest.")
        print("     This is expected — the key finding is that a REAL LLM (Qwen) reasoning")
        print("     over the semantic descriptions should far exceed a simple rule tree.")
    
    print()
    print("  📝 Next Steps:")
    if not use_claude:
        print("     1. Re-run with --use-claude to benchmark the real LLM.")
    print("     2. Enrich semantic translator with MFCC, chroma, onset features (Phase 3).")
    print("     3. Deploy Qwen on AWS/GCP and swap in for Claude (Phase 4).")
    print("     4. Begin collecting RLHF data from user confirmations in the mobile app.")
    print("=" * 70)
    
    # Save results as JSON for the validation phase
    results = {
        "llm_backend": llm_label,
        "llm_accuracy": round(llm_accuracy, 4),
        "llm_n_samples": len(y_llm_eval),
        "rf_accuracy": round(rf_accuracy, 4),
        "rf_n_samples": len(y_ml_test),
        "dataset_size": len(df),
        "n_subgenres": df['subgenre'].nunique(),
        "top_10_feature_importances": {f: round(float(i), 4) for f, i in importances},
        "sample_descriptions": [
            {"actual": sp["actual"], "predicted": sp["predicted"], "description": sp["description"]}
            for sp in sample_prompts[:2]
        ],
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/prototype_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to data/prototype_results.json")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Subgenre Classification Prototype")
    parser.add_argument("--dataset",    type=str,  default=None,
                        help="Path to Beatport CSV. Omit to use synthetic data.")
    parser.add_argument("--use-claude", action="store_true",
                        help="Use real Claude CLI calls instead of the rule-based simulator.")
    parser.add_argument("--llm-sample", type=int,  default=3,
                        help="Tracks per subgenre to send to Claude (default: 3).")
    args = parser.parse_args()

    run_prototype(
        dataset_path=args.dataset,
        use_claude=args.use_claude,
        llm_sample=args.llm_sample,
    )
