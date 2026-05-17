import librosa
import numpy as np
import json
import warnings
import scipy.io.wavfile as wavf

warnings.filterwarnings('ignore')

def generate_dummy_audio(filename="dummy_track.wav", duration=5.0, sr=22050):
    """Generates a dummy 4-on-the-floor kick drum pattern to test feature extraction."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 128 BPM -> 2.13 beats per second. Let's just make a pulse every 0.468 seconds.
    beat_interval = 60.0 / 128.0
    audio = np.zeros_like(t)
    
    for i in np.arange(0, duration, beat_interval):
        idx = int(i * sr)
        end_idx = min(idx + int(0.1 * sr), len(t))
        if idx < len(t):
            # A low frequency sine wave for a "kick"
            kick_t = np.linspace(0, 0.1, end_idx - idx, endpoint=False)
            audio[idx:end_idx] = np.sin(2 * np.pi * 60 * kick_t) * np.exp(-15 * kick_t)
            
    wavf.write(filename, sr, (audio * 32767).astype(np.int16))
    return filename

def extract_features(audio_path):
    """Extracts key electronic music features using Librosa."""
    print(f"Loading {audio_path}...")
    y, sr = librosa.load(audio_path, sr=None)
    
    # 1. BPM / Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    
    # 2. Spectral Centroid (brightness)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = float(np.mean(cent))
    
    # 3. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    mean_rolloff = float(np.mean(rolloff))
    
    # 4. Zero Crossing Rate (percussive vs tonal)
    zcr = librosa.feature.zero_crossing_rate(y)
    mean_zcr = float(np.mean(zcr))
    
    features = {
        "bpm": round(bpm, 2),
        "spectral_centroid_hz": round(mean_centroid, 2),
        "spectral_rolloff_hz": round(mean_rolloff, 2),
        "zero_crossing_rate": round(mean_zcr, 4)
    }
    
    return features

if __name__ == "__main__":
    dummy_file = generate_dummy_audio()
    print("Generated dummy audio file.")
    features = extract_features(dummy_file)
    print("Extracted Features:")
    print(json.dumps(features, indent=2))
