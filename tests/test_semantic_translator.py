import pytest
from src.semantic_translator import (
    describe_tempo, describe_spectral_centroid, describe_zcr,
    translate_features_to_description
)


class TestDescribeTempo:
    def test_ambient_range(self):
        desc = describe_tempo(75)
        assert "very slow" in desc
        assert "ambient" in desc.lower()

    def test_house_range(self):
        desc = describe_tempo(128)
        assert "standard dance" in desc

    def test_dnb_range(self):
        desc = describe_tempo(170)
        assert "very fast" in desc
        assert "drum and bass" in desc.lower()

    def test_hardstyle_range(self):
        desc = describe_tempo(150)
        assert "fast tempo" in desc


class TestDescribeSpectralCentroid:
    def test_dark_bass(self):
        desc = describe_spectral_centroid(500)
        assert "dark" in desc or "bass" in desc

    def test_bright(self):
        desc = describe_spectral_centroid(4500)
        assert "bright" in desc


class TestDescribeZcr:
    def test_tonal(self):
        desc = describe_zcr(0.01)
        assert "tonal" in desc

    def test_percussive(self):
        desc = describe_zcr(0.16)
        assert "percussive" in desc


class TestTranslateFeatures:
    def test_full_feature_set(self):
        features = {
            "bpm": 128,
            "spectral_centroid_hz": 3000,
            "spectral_rolloff_hz": 5000,
            "zero_crossing_rate": 0.08,
            "energy": 0.07,
            "mfcc_std": 15
        }
        desc = translate_features_to_description(features)
        assert isinstance(desc, str)
        assert len(desc) > 50
        assert "128" in desc

    def test_partial_features(self):
        features = {"bpm": 140}
        desc = translate_features_to_description(features)
        assert "140" in desc
        # Should not crash with missing keys

    def test_empty_features(self):
        desc = translate_features_to_description({})
        assert desc == ""
