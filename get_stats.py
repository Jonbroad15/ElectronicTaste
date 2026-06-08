import json
import re
import urllib.request
import sys
import os

# Append scripts to path so we can import download_raveform
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from download_raveform import fetch_manifest, load_taxonomy, build_labels, filter_manifest

manifest_url = "https://raw.githubusercontent.com/mir-aidj/djmix-dataset/main/dataset/djmix-dataset.json"
taxonomy_url = "https://raw.githubusercontent.com/Mendiak/pulse.roots/main/data/pulseroots.genres.json"

print("Loading taxonomy...")
tag_to_genre, taxonomy_index = load_taxonomy(taxonomy_url)

print("Fetching manifest...")
entries = fetch_manifest(manifest_url)

print("Filtering manifest...")
selected = filter_manifest(entries, tag_to_genre, set())

total_seconds = 0
total_labelled_samples = 0
class_counts = {}

def get_mix_duration(entry):
    # Extract duration from the last track in the tracklist
    tracklist = entry.get('tracklist', [])
    max_min = 0
    for t in tracklist:
        title = t.get('title', '')
        m = re.search(r'\[(\d+)\]', title)
        if m:
            val = int(m.group(1))
            if val > max_min:
                max_min = val
    if max_min == 0:
        # Default fallback if no timestamps: guess 60 mins
        return 3600
    # duration is max_min minutes + 5 minutes for the last track approx
    return (max_min + 5) * 60

for entry in selected:
    duration_sec = get_mix_duration(entry)
    total_seconds += duration_sec
    
    # number of 30s samples
    samples = int(duration_sec / 30)
    total_labelled_samples += samples
    
    mix_labels = build_labels(entry.get("tags", []), tag_to_genre, taxonomy_index)
    
    # count L1 genres
    for g in mix_labels["l1_genres"]:
        class_counts[g] = class_counts.get(g, 0) + samples

print("Total selected mixes:", len(selected))
print("Total duration (seconds):", total_seconds)
print("Total labelled samples (30s):", total_labelled_samples)
print("Class counts (L1, in samples):")
for k, v in class_counts.items():
    print(f"  {k}: {v}")

