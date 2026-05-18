import os
import pytest
from src.extract_features import generate_dummy_audio, extract_features

def test_generate_dummy_audio():
    """Test that the dummy audio generator actually creates a file."""
    filename = "test_dummy.wav"
    try:
        output_file = generate_dummy_audio(filename=filename, duration=2.0)
        assert os.path.exists(output_file)
        assert output_file == filename
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def test_extract_features_structure():
    """Test that extract_features returns the expected dictionary structure."""
    filename = "test_dummy_features.wav"
    try:
        generate_dummy_audio(filename=filename, duration=2.0)
        features = extract_features(filename)
        
        # Verify it's a dictionary
        assert isinstance(features, dict)
        
        # Verify the expected keys are present
        expected_keys = ["bpm", "spectral_centroid_hz", "spectral_rolloff_hz", "zero_crossing_rate"]
        for key in expected_keys:
            assert key in features
            
        # Verify values are floats
        for key in expected_keys:
            assert isinstance(features[key], float)
            
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def test_extract_features_values():
    """Test that extract_features returns reasonable values for our specific dummy sine wave."""
    filename = "test_dummy_values.wav"
    try:
        generate_dummy_audio(filename=filename, duration=5.0)
        features = extract_features(filename)
        
        # The dummy track is generated with pulses at 128 BPM intervals.
        # Librosa might predict roughly 128 or a multiple/fraction.
        # We'll assert it's somewhat close to 128 BPM.
        assert 120 <= features["bpm"] <= 140
        
        # Since it's a low frequency 60Hz kick, the spectral centroid should be very low.
        assert features["spectral_centroid_hz"] < 200
        
        # Zero crossing rate should be very low since it's mostly a smooth sine wave and silence.
        assert features["zero_crossing_rate"] < 0.05
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)
