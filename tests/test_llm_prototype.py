import pytest
from src.llm_prototype import (
    generate_synthetic_dataset, rule_based_llm_classifier, 
    build_llm_prompt, SUBGENRE_PROFILES
)
from src.semantic_translator import translate_features_to_description


class TestSyntheticDataset:
    def test_generates_correct_row_count(self):
        df = generate_synthetic_dataset(n_per_genre=10)
        assert len(df) == 10 * len(SUBGENRE_PROFILES)

    def test_contains_required_columns(self):
        df = generate_synthetic_dataset(n_per_genre=5)
        required = ['bpm', 'spectral_centroid_hz', 'spectral_rolloff_hz',
                     'zero_crossing_rate', 'energy', 'mfcc_std', 'subgenre']
        for col in required:
            assert col in df.columns

    def test_all_subgenres_present(self):
        df = generate_synthetic_dataset(n_per_genre=5)
        for genre in SUBGENRE_PROFILES.keys():
            assert genre in df['subgenre'].values

    def test_bpm_within_expected_ranges(self):
        df = generate_synthetic_dataset(n_per_genre=20, seed=99)
        dnb = df[df['subgenre'] == 'Drum and Bass']
        assert dnb['bpm'].min() >= 159  # Should be close to 160
        assert dnb['bpm'].max() <= 181


class TestRuleBasedClassifier:
    def test_ambient_classification(self):
        features = {"bpm": 75, "spectral_centroid_hz": 1000, 
                     "zero_crossing_rate": 0.02, "energy": 0.01}
        desc = translate_features_to_description(features)
        prediction = rule_based_llm_classifier(desc)
        assert prediction == "Ambient"

    def test_dnb_classification(self):
        features = {"bpm": 172, "spectral_centroid_hz": 4000, 
                     "zero_crossing_rate": 0.10, "energy": 0.08}
        desc = translate_features_to_description(features)
        prediction = rule_based_llm_classifier(desc)
        assert prediction == "Drum and Bass"


class TestPromptBuilder:
    def test_prompt_contains_description(self):
        desc = "The track has a standard dance tempo of 128 BPM."
        prompt = build_llm_prompt(desc, ["House", "Techno"])
        assert "128 BPM" in prompt
        assert "House" in prompt
        assert "Techno" in prompt

    def test_prompt_contains_instructions(self):
        prompt = build_llm_prompt("test description", ["House"])
        assert "Prediction" in prompt
        assert "Instructions" in prompt
