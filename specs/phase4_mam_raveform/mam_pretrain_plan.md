# Masked Audio Modeling (MAM) Pre-training Plan

This document outlines the strategy and setup for continuing the self-supervised pre-training of the MERT-v1-95M encoder on the Raveform dataset using Masked Audio Modeling.

---

## 1. Objectives

* **Domain Adaptation:** Adapt the general-purpose MERT music representations to the specific characteristics of continuous, DJ-curated electronic dance music (EDM) mixes.
* **Feature Resolution:** Improve the encoder's ability to discriminate fine-grained subgenres (e.g. distinguishing Melodic Techno from Progressive House), which currently shows low baseline performance (26.9%).
* **SSL Continuation:** Leverage self-supervised learning on unlabelled long-form audio sets without human annotation costs.

---

## 2. Pre-training Task & Loss

MERT is pre-trained using a Masked Audio Modeling (MAM) framework with a BERT-style encoder and dual teacher models:

1. **Acoustic Teacher:** An RVQ-VAE (Residual Vector Quantizer Video-Audio Encoder) that provides discrete acoustic codes for each audio frame.
2. **Musical Teacher:** Constant-Q Transform (CQT) features that capture pitch, frequency, and harmonic structure.

### Training Objective
* Mask a subset of the input frame-level embeddings (typically 40%–50% of frames).
* Train the MERT transformer layers to predict the acoustic codes and CQT features of the masked frames.
* **Loss Function:** Cross-entropy loss for the discrete acoustic codes + mean squared error (MSE) for the CQT feature reconstruction.

---

## 3. Dataset (Raveform)

* **Source:** 5,040 long-form electronic music mixes from `mir-aidj/djmix-dataset` (SoundCloud, Mixcloud, YouTube).
* **Format:** Mono WAV, 24 kHz, peak-normalized.
* **Volume:** ~500–700 GB of audio data.
* **Chunking:** Mixes are split into 5-second segments (overlap of 1s for data augmentation), yielding millions of training windows.

---

## 4. Hyperparameters & Configuration

| Parameter | Value |
|---|---|
| **Base Model** | `m-a-p/MERT-v1-95M` (94.4M parameters) |
| **Optimizer** | `AdamW` ($\beta_1=0.9, \beta_2=0.98$, weight decay 0.01) |
| **Learning Rate** | $1 \times 10^{-4}$ with linear warmup and cosine decay |
| **Warmup Steps** | 3,000 steps |
| **Total Steps** | 50,000 steps (approx. 5–6 epochs over the Raveform dataset) |
| **Batch Size** | 64 (accumulated over 4 steps if GPU memory is limited) |
| **Mask Probability** | 0.5 (masking duration of 10-20 consecutive frames) |

---

## 5. Implementation Steps on GCP GPU Instance

1. **Environment Setup:** Launch `electronic-taste-train` with PyTorch GPU image and attach the 2 TB persistent disk `/mnt/data/` containing the Raveform dataset.
2. **Feature Extraction Setup:** Initialize the RVQ-VAE quantizer and CQT feature extraction teachers on the GPU.
3. **Data Loading:** Implement a streaming PyTorch `IterableDataset` to feed chunks of mixes dynamically from the disk without loading the entire 2 TB into RAM.
4. **MAM Training Loop:** Run the training loop utilizing PyTorch mixed-precision (`torch.cuda.amp`) and save checkpoints every 5,000 steps to `/mnt/data/models/mam_pretrain/`.
5. **Downstream Classifier Evaluation:** After pre-training, freeze the new MERT weights and re-train the linear classification head on the target EDM subgenre labels to evaluate the accuracy boost.
