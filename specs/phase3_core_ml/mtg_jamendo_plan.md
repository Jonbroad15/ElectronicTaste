# MTG-Jamendo EDM Fine-Tuning Plan (M5)

> Phase 3 Step 9 — extends the GTZAN pipeline to 20-class EDM subgenre classification.

---

## 20-Class Taxonomy

The taxonomy (`data/taxonomy.json`) covers the main lineages of electronic music:

| Lineage | Classes |
|---|---|
| Techno / industrial | techno, minimal, melodic techno, hardcore, gabber, industrial, acid |
| House | house, deep house, tech house, progressive house |
| Dance / rave | trance, drum and bass, dubstep, jungle, breakbeat, garage, electro |
| Mood / era | ambient, synthwave |

**Rationale**: Classes were chosen to (a) be unambiguous enough for a listener to agree on a single label, (b) be well-represented in MTG-Jamendo's freeform tag vocabulary, and (c) cover the EDM subgenres most relevant to the app's target audience. Overlapping styles (e.g. "melodic techno house") are mapped to the dominant canonical class via `tag_map`.

---

## Expected Class Distribution

MTG-Jamendo contains ~16K electronic tracks (raw_30s_cleantags_50artists subset). Based on tag frequencies in the published metadata:

| Class | Est. tracks |
|---|---|
| techno | 1,500–2,500 |
| house | 1,200–2,000 |
| ambient | 1,000–2,000 |
| trance | 800–1,500 |
| progressive house | 400–800 |
| deep house | 300–700 |
| drum and bass | 300–600 |
| electro | 200–500 |
| synthwave | 200–500 |
| industrial | 200–400 |
| breakbeat | 150–350 |
| dubstep | 100–300 |
| tech house | 100–300 |
| acid | 100–250 |
| melodic techno | 100–200 |
| minimal | 100–200 |
| garage | 80–200 |
| jungle | 80–150 |
| hardcore | 60–150 |
| gabber | 50–100 |

Classes with < 50 verified samples are flagged by `prepare_mtg_jamendo.py` and should be excluded or supplemented before training.

---

## Training Plan

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 50 | More classes than GTZAN (20 vs 10); more training needed |
| Learning rate | 1e-3 | Consistent with GTZAN run; cosine annealing decay |
| Batch size | 32 | Fits comfortably in 16 GB unified memory (embeddings only) |
| Train / val split | 80 / 20 | Standard; stratified by class when possible |
| Optimizer | AdamW, weight_decay=1e-4 | Same as GTZAN run |
| Scheduler | CosineAnnealingLR (T_max=50) | Smooth LR decay |
| Encoder | MERT-v1-95M, **frozen** | Avoids overfitting on limited data |
| Head architecture | LayerNorm → Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→20) | Same as GTZAN |
| Audio clip | 5 s @ 24 kHz | MERT's expected input |

---

## NFR Target

**≥ 60% top-1 accuracy** on the 20-class EDM benchmark (held-out val set).

Baseline expectation: GTZAN achieved 93% on 10 classes with fine-tuning. EDM subgenres are more acoustically similar, so 60% is the conservative floor; 70%+ is achievable with sufficient data per class.

---

## End-to-End Pipeline

```bash
# 1. Build manifest (no audio download)
python scripts/download_mtg_jamendo.py

# 2. Check class distribution / warnings
python scripts/prepare_mtg_jamendo.py

# 3. Download audio (takes 2–6 hours for ~16K tracks)
python scripts/download_mtg_jamendo.py --download-audio

# 4. Re-verify after download
python scripts/prepare_mtg_jamendo.py

# 5. Fine-tune classifier
python -m src.training.train \
    --data-dir data/mtg_jamendo \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3

# 6. Benchmark
python src/scripts/benchmark.py \
    --data-dir data/mtg_jamendo \
    --model models/classifier_best.pt
```
