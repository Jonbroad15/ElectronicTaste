"""
LLM Reasoning Prototype for EDM Subgenre Classification
========================================================
Phase 2, Step 3: Test whether an LLM can classify electronic music
subgenres from structured audio feature descriptions.

This prototype:
1. Generates a synthetic Beatport-style dataset (or loads a real one)
2. Applies the Semantic Translation Layer to convert features to text
3. Tests a rule-based "LLM simulator" to establish a baseline
4. Compares against a traditional ML classifier (Random Forest)
5. Outputs accuracy metrics for both approaches

To use with the REAL Kaggle Beatport dataset:
  1. Download from: https://www.kaggle.com/datasets/caparrini/electronic-music-features-201611-beatporttop100
  2. Place the CSV file at: data/beatport_top100.csv
  3. Run: python src/llm_prototype.py --dataset data/beatport_top100.csv
"""

import argparse
import json
import os
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
    Loads the real Beatport Kaggle dataset CSV.
    Maps its columns to our standardised feature names.
    """
    df = pd.read_csv(csv_path)
    
    # The Beatport CSVs use pyAudioAnalysis feature names.
    # We map the most relevant ones to our standard names.
    column_mapping = {}
    
    # Try to find the subgenre/category column (usually the last few columns)
    possible_genre_cols = [c for c in df.columns if 'genre' in c.lower() or 'category' in c.lower() or 'class' in c.lower()]
    if possible_genre_cols:
        column_mapping[possible_genre_cols[0]] = 'subgenre'
    
    # Detect BPM column
    bpm_cols = [c for c in df.columns if 'bpm' in c.lower() or 'tempo' in c.lower()]
    if bpm_cols:
        column_mapping[bpm_cols[0]] = 'bpm'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # If we can't find mapped features, use the first N numeric columns as raw features
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns[:10])}... (truncated)")
    
    if 'subgenre' in df.columns:
        print(f"Subgenres found: {df['subgenre'].nunique()} unique values")
        print(f"Distribution:\n{df['subgenre'].value_counts().head(10)}")
    
    return df


def rule_based_llm_classifier(description):
    """
    Simulates how an LLM would reason about a track's subgenre given a
    natural language description of its audio features.
    
    In production, this would be replaced with an actual LLM API call
    (e.g., Qwen hosted on AWS/GCP). This rule-based version tests whether
    the semantic translation layer provides enough signal for classification.
    """
    desc_lower = description.lower()
    
    # Decision tree mimicking LLM reasoning chains.
    # The semantic translator embeds genre hints in tempo descriptions,
    # so we check for those keywords first.
    
    # ── Extreme tempo ranges (unambiguous) ──
    if "extremely fast" in desc_lower:
        return "Drum and Bass"
    
    if "very fast" in desc_lower:
        # The translator says "drum and bass or jungle" for 160-175 BPM
        if "drum and bass" in desc_lower or "jungle" in desc_lower:
            return "Drum and Bass"
        return "Hardstyle"
    
    if "fast tempo" in desc_lower:
        # 145-160 BPM range: hardstyle, psytrance, hard techno
        if "psytrance" in desc_lower:
            return "Trance"
        if "hardstyle" in desc_lower:
            return "Hardstyle"
        return "Hardstyle"
    
    if "very slow" in desc_lower:
        return "Ambient"
    
    if "slow-to-moderate" in desc_lower:
        if "dark" in desc_lower or "bass-heavy" in desc_lower:
            return "Ambient"
        return "Deep House"
    
    # ── Elevated tempo: 135-145 BPM ──
    if "elevated tempo" in desc_lower:
        if "bright" in desc_lower or "harsh" in desc_lower:
            return "Techno"
        if "percussive" in desc_lower:
            return "Tech House"
        return "Trance"
    
    # ── Standard dance tempo: 125-135 BPM ──
    if "standard dance tempo" in desc_lower:
        if "bright" in desc_lower and "percussive" in desc_lower:
            return "Electro House"
        if "dark" in desc_lower or "bass-heavy" in desc_lower:
            return "Dubstep"
        if "balanced" in desc_lower and "percussive" in desc_lower:
            return "Techno"
        if "warm" in desc_lower:
            if "tonal" in desc_lower and "smooth" in desc_lower:
                return "Deep House"
            return "Progressive House"
        return "Trance"
    
    # ── Moderate tempo: 110-125 BPM ──
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
    
    # Fallback
    return "House"


def build_llm_prompt(description, subgenre_list):
    """
    Constructs the full prompt that would be sent to an LLM like Qwen.
    This is saved for reference / future API integration.
    """
    genres_str = ", ".join(subgenre_list)
    
    prompt = f"""You are an expert electronic music classifier. Given the following audio analysis of a track, predict which electronic music subgenre it belongs to.

## Audio Analysis
{description}

## Possible Subgenres
{genres_str}

## Instructions
1. Consider the tempo (BPM) as the primary differentiator.
2. Use spectral characteristics to distinguish between subgenres with similar tempos.
3. Consider the percussive vs tonal balance.
4. Return ONLY the subgenre name, nothing else.

## Prediction:"""
    
    return prompt


def run_prototype(dataset_path=None):
    """Main prototype execution."""
    print("=" * 70)
    print("  Electronic Taste — LLM Reasoning Model Prototype")
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
    
    # Ensure we have the features we need
    feature_cols = ['bpm', 'spectral_centroid_hz', 'spectral_rolloff_hz', 
                    'zero_crossing_rate', 'energy', 'mfcc_std']
    available_features = [c for c in feature_cols if c in df.columns]
    
    if len(available_features) < 2:
        print("\n⚠️  Not enough recognized feature columns. Using all numeric columns for ML baseline.")
        available_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'subgenre' in available_features:
            available_features.remove('subgenre')
    
    # ── Step 2: Split into train/test ──
    X = df[available_features]
    y = df['subgenre']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split: {len(X_train)} train, {len(X_test)} test")
    
    # ── Step 3: Semantic Translation + Rule-Based LLM Classifier ──
    print("\n" + "─" * 70)
    print("  TEST A: Semantic Translation → LLM Reasoning (Rule-Based Simulation)")
    print("─" * 70)
    
    llm_predictions = []
    sample_prompts = []
    
    for idx, row in X_test.iterrows():
        features = row.to_dict()
        description = translate_features_to_description(features)
        prediction = rule_based_llm_classifier(description)
        llm_predictions.append(prediction)
        
        # Save a few sample prompts for inspection
        if len(sample_prompts) < 3:
            prompt = build_llm_prompt(description, list(SUBGENRE_PROFILES.keys()))
            sample_prompts.append({
                "actual": y_test.iloc[len(sample_prompts)],
                "predicted": prediction,
                "description": description,
                "prompt": prompt
            })
    
    llm_accuracy = accuracy_score(y_test, llm_predictions)
    print(f"\n✅ LLM Simulator Accuracy: {llm_accuracy:.1%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, llm_predictions, zero_division=0))
    
    # ── Step 4: Traditional ML Baseline (Random Forest) ──
    print("─" * 70)
    print("  TEST B: Traditional ML Baseline (Random Forest)")
    print("─" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15)
    rf.fit(X_train_scaled, y_train)
    rf_predictions = rf.predict(X_test_scaled)
    
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"\n✅ Random Forest Accuracy: {rf_accuracy:.1%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, rf_predictions, zero_division=0))
    
    # ── Step 5: Feature Importance (from RF) ──
    print("─" * 70)
    print("  Feature Importance (Random Forest)")
    print("─" * 70)
    importances = sorted(
        zip(available_features, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {feat:30s} {imp:.3f} {bar}")
    
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
        print("  ⚠️  The LLM reasoning approach needs improvement.")
        print("     Consider enriching the feature set or improving the semantic translations.")
    
    print()
    print("  📝 Next Steps:")
    print("     1. Download the real Kaggle Beatport dataset and re-run this prototype.")
    print("     2. Replace rule_based_llm_classifier() with actual Qwen API calls.")
    print("     3. Begin collecting RLHF data from user confirmations.")
    print("=" * 70)
    
    # Save results as JSON for the validation phase
    results = {
        "llm_accuracy": round(llm_accuracy, 4),
        "rf_accuracy": round(rf_accuracy, 4),
        "dataset_size": len(df),
        "n_subgenres": df['subgenre'].nunique(),
        "feature_importance": {f: round(float(i), 4) for f, i in importances},
        "sample_prompts": sample_prompts[:2],
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/prototype_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to data/prototype_results.json")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Subgenre Classification Prototype")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to a Beatport CSV dataset. If not provided, uses synthetic data.")
    args = parser.parse_args()
    
    run_prototype(dataset_path=args.dataset)
