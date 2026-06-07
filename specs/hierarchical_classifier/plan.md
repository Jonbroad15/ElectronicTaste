# Implementation Plan: Hierarchical EDM Classifier

> References: [architecture_ideas.md](../phase4_mam_raveform/architecture_ideas.md), [requirements.md](requirements.md)

---

## Directory Layout

```
src/
  training/
    train.py                 # Update to use new architecture args
    loss_functions.py        # New: DAG-aware loss and SupCon
    contrastive_model.py     # New: LoRA MERT + Chunking + Contrastive Head
```

---

## Step-by-Step Plan

### Step 1 — Implement Contrastive LoRA MERT Model

1. Define `ContrastiveMERTClassifier` in `src/models/contrastive_model.py`.
2. Integrate `peft` (LoRA) into the 95M MERT encoder.
3. Implement the 5s chunking and pooling aggregation logic.
4. Add the projection head for the embedding space.

### Step 2 — Implement Loss Functions

1. Implement `MultiLabelSupConLoss` in `src/training/loss_functions.py`.
2. Implement the DAG-aware soft-margin hierarchical penalty.

### Step 3 — Update Training Script & GCP Setup

1. Modify `src/training/train.py` to instantiate the `ContrastiveMERTClassifier` and use the new loss functions.
2. Ensure prototypes are computed and saved at the end of the training epoch.
3. Update `scripts/gcp_setup_training.sh` to pass the correct arguments (e.g., `--use-lora`, `--loss contrastive`).

### Step 4 — Inference & Few-Shot Module

1. Implement `inference.py` using Independent Thresholding against the saved Prototypes.
2. Implement a `add_new_subgenre()` function to dynamically compute a new prototype and update the class registry without retraining.

---

## Dependencies

Add to `requirements.txt`:

```
peft>=0.5.0
```

---

## Milestones

| # | Deliverable | Done when |
|---|---|---|
| M1 | LoRA Model & Loss | Model compiles, runs a forward pass, and computes loss without OOM on 30s samples. |
| M2 | Training Pipeline | GCP L4 instance trains the contrastive space successfully and saves prototypes. |
| M3 | Few-Shot Inference | We can dynamically add a new genre and correctly classify an unseen track. |
