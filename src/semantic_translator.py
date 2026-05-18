"""
Semantic Translation Layer
==========================
Converts raw numeric audio features into natural language descriptions
that an LLM can reason about effectively.

This addresses the ML Specialist's deep review recommendation:
"Instead of {"bpm": 130, "zcr": 0.8}, the prompt should read:
'The track has a fast tempo of 130 BPM, is highly percussive, and
features bright, noisy synth elements.'"
"""


def describe_tempo(bpm):
    """Translate BPM into a human-readable tempo description."""
    if bpm < 90:
        return f"a very slow tempo of {bpm:.0f} BPM, suggesting downtempo or ambient music"
    elif bpm < 110:
        return f"a slow-to-moderate tempo of {bpm:.0f} BPM, typical of deep house or lo-fi"
    elif bpm < 125:
        return f"a moderate tempo of {bpm:.0f} BPM, common in house and disco-influenced genres"
    elif bpm < 135:
        return f"a standard dance tempo of {bpm:.0f} BPM, typical of house, techno, and trance"
    elif bpm < 145:
        return f"an elevated tempo of {bpm:.0f} BPM, common in tech house, hard house, or uplifting trance"
    elif bpm < 160:
        return f"a fast tempo of {bpm:.0f} BPM, suggesting hard techno, psytrance, or hardstyle"
    elif bpm < 175:
        return f"a very fast tempo of {bpm:.0f} BPM, typical of drum and bass or jungle"
    else:
        return f"an extremely fast tempo of {bpm:.0f} BPM, suggesting hardcore, gabber, or speedcore"


def describe_spectral_centroid(centroid_hz):
    """Translate spectral centroid into brightness/timbre description."""
    if centroid_hz < 1000:
        return "very dark and bass-heavy timbres with minimal high-frequency content"
    elif centroid_hz < 2000:
        return "warm, mid-range-focused timbres with moderate brightness"
    elif centroid_hz < 3500:
        return "balanced timbres with a mix of low-end warmth and high-end presence"
    elif centroid_hz < 5000:
        return "bright timbres with prominent high-frequency synthesizer elements"
    else:
        return "very bright and harsh timbres, suggesting aggressive or noisy production"


def describe_spectral_rolloff(rolloff_hz):
    """Translate spectral rolloff into frequency content description."""
    if rolloff_hz < 2000:
        return "Most of the sonic energy is concentrated in the low frequencies (sub-bass and bass)"
    elif rolloff_hz < 4000:
        return "The sonic energy is spread across low and mid frequencies"
    elif rolloff_hz < 6000:
        return "The track has a full frequency spectrum with energy extending into the upper mids"
    else:
        return "The track has significant high-frequency energy, suggesting bright pads, cymbals, or noise elements"


def describe_zcr(zcr):
    """Translate zero-crossing rate into percussive vs tonal description."""
    if zcr < 0.02:
        return "highly tonal and smooth, dominated by sustained bass or pads"
    elif zcr < 0.05:
        return "mostly tonal with some percussive transients"
    elif zcr < 0.10:
        return "a balanced mix of tonal and percussive elements"
    elif zcr < 0.15:
        return "moderately percussive with sharp transients"
    else:
        return "highly percussive and noisy, dominated by drums, clicks, or distortion"


def describe_energy(energy):
    """Translate RMS energy into loudness/dynamics description."""
    if energy < 0.01:
        return "The track is very quiet and dynamic, suggesting ambient or minimal music"
    elif energy < 0.05:
        return "The track has moderate loudness with preserved dynamics"
    elif energy < 0.10:
        return "The track is loud and compressed, typical of club-ready dance music"
    else:
        return "The track is very loud and heavily compressed, maximizing energy for peak-time sets"


def describe_mfcc_spread(mfcc_std):
    """Translate MFCC standard deviation into timbral variety description."""
    if mfcc_std < 10:
        return "The timbral palette is narrow and consistent, suggesting repetitive or minimalist production"
    elif mfcc_std < 20:
        return "The timbral palette is moderately varied"
    else:
        return "The timbral palette is wide and varied, suggesting complex arrangements with many different sound sources"


def translate_features_to_description(features):
    """
    Takes a dictionary of numeric audio features and produces a natural
    language description suitable for LLM reasoning.
    
    Args:
        features: dict with keys like 'bpm', 'spectral_centroid_hz', etc.
    
    Returns:
        A multi-sentence natural language description of the track.
    """
    parts = []
    
    if 'bpm' in features:
        parts.append(f"The track has {describe_tempo(features['bpm'])}.")
    
    if 'spectral_centroid_hz' in features:
        parts.append(f"The track features {describe_spectral_centroid(features['spectral_centroid_hz'])}.")
    
    if 'spectral_rolloff_hz' in features:
        parts.append(f"{describe_spectral_rolloff(features['spectral_rolloff_hz'])}.")

    if 'zero_crossing_rate' in features:
        parts.append(f"The sonic character is {describe_zcr(features['zero_crossing_rate'])}.")
    
    if 'energy' in features:
        parts.append(describe_energy(features['energy']))
    
    if 'mfcc_std' in features:
        parts.append(describe_mfcc_spread(features['mfcc_std']))

    return " ".join(parts)
