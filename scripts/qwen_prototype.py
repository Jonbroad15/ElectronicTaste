#!/usr/bin/env python3
"""
Qwen2-Audio Genre Classification Prototype
==========================================

Uses Qwen2-Audio-7B-Instruct to classify music genres zero-shot.
Since it's a 7B parameter model, we test it on a small subset of the
GTZAN dataset to evaluate accuracy and inference speed on Mac Apple Silicon.

Architecture:
  Audio WAV + Text Prompt → Qwen2-Audio → Text Output (Genre Name)

Usage: python scripts/qwen_prototype.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DATA_DIR = Path("data/gtzan_audio")
RESULTS_FILE = Path("data/qwen_prototype_results.json")
SAMPLES_PER_GENRE = 2  # Evaluate on a tiny subset to save time/memory

GENRE_NAMES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ─── Main Pipeline ───────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def collect_test_files(samples_per_genre=SAMPLES_PER_GENRE):
    files = []
    labels = []
    
    random.seed(42)
    for genre in GENRE_NAMES:
        genre_dir = DATA_DIR / genre
        if not genre_dir.exists():
            continue
            
        genre_files = list(genre_dir.glob("*.wav"))
        selected = random.sample(genre_files, min(len(genre_files), samples_per_genre))
        files.extend(selected)
        labels.extend([genre] * len(selected))
        
    return files, labels

def main():
    print("=" * 60)
    print("Qwen2-Audio Genre Classification Prototype (Zero-Shot)")
    print("=" * 60)

    device = get_device()
    print(f"\nDevice: {device}")
    
    files, labels = collect_test_files()
    print(f"Testing on {len(files)} files ({SAMPLES_PER_GENRE} per genre)")
    
    if not files:
        print("ERROR: No files found in data/gtzan_audio/")
        sys.exit(1)

    print(f"\nStep 1: Loading model {MODEL_ID}...")
    start_time = time.time()
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto", 
        torch_dtype=torch.float16  # Use fp16 to fit in memory
    )
    
    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.1f}s")
    
    results = []
    correct_count = 0
    total_time = 0
    
    prompt = f"What is the music genre of this audio clip? Respond with exactly one word from this list: {', '.join(GENRE_NAMES)}."
    
    print("\nStep 2: Running Inference...")
    for i, (filepath, true_genre) in enumerate(zip(files, labels)):
        # Load audio (Qwen2-Audio uses 16kHz)
        audio, sr = librosa.load(filepath, sr=16000)
        
        # Take just the first 10 seconds to speed up processing
        # and stay within model context limits
        audio = audio[:16000 * 10] 
        
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "dummy_url"}, # The processor will use the actual audio array
                {"type": "text", "text": prompt}
            ]}
        ]
        
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        inputs = processor(
            text=text_prompt, 
            audios=audio, 
            return_tensors="pt", 
            sampling_rate=sr
        ).to(model.device)
        
        # Generate
        start_inf = time.time()
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=10)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        inf_time = time.time() - start_inf
        total_time += inf_time
        
        predicted = response.strip().lower()
        # Clean up output just in case
        for g in GENRE_NAMES:
            if g in predicted:
                predicted = g
                break
                
        is_correct = (predicted == true_genre)
        if is_correct:
            correct_count += 1
            
        print(f"  [{i+1}/{len(files)}] True: {true_genre:<10} | Pred: {predicted:<15} | {'✅' if is_correct else '❌'} | {inf_time:.1f}s")
        
        results.append({
            "file": str(filepath.name),
            "true_genre": true_genre,
            "predicted_genre": predicted,
            "raw_response": response.strip(),
            "correct": is_correct,
            "inference_time": inf_time
        })

    accuracy = correct_count / len(files)
    avg_inf_time = total_time / len(files)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Model:         {MODEL_ID}")
    print(f"  Test Accuracy: {accuracy:.1%} ({correct_count}/{len(files)})")
    print(f"  Avg Inference: {avg_inf_time:.2f}s per file")
    
    summary = {
        "model": MODEL_ID,
        "test_size": len(files),
        "accuracy": accuracy,
        "avg_inference_time": avg_inf_time,
        "results": results
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
