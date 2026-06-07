# Requirements: Hierarchical EDM Classifier

## Goal

Build a multi-label, hierarchical classifier for EDM audio samples using a MERT backbone adapted with LoRA, utilizing Supervised Contrastive Learning to support few-shot learning of new subgenres.

---

## Functional Requirements

### FR-1: Hierarchical Multi-Label Prediction
- The model must output probabilities for L1, L2, and L3 subgenres simultaneously.
- A single audio sample must be able to trigger multiple labels at the same level (e.g., both "House" and "Techno").

### FR-2: Feature Extraction and Temporal Aggregation
- Use the 95M MERT model.
- Apply Parameter-Efficient Fine-Tuning (LoRA) to the attention layers of the MERT encoder.
- Chunk the 30-second audio inputs into 5-second segments, pool the segment representations, and aggregate them to form a clip-level embedding.

### FR-3: Few-Shot Class Incremental Learning (FSCIL)
- The model must use Supervised Contrastive Learning (SupCon) to learn a robust embedding space.
- Subgenres must be represented by Prototype vectors (mean embeddings).
- New subgenres must be addable without immediate retraining by computing new Prototypes from few-shot examples.

### FR-4: Hierarchical Constraints
- The loss function must enforce ontology paths using a DAG-aware loss with soft margins (hierarchical label smoothing) to prevent penalizing valid rare combinations.

---

## Non-Functional Requirements

### NFR-1: Compute Efficiency
- Downstream training must fit on a single NVIDIA L4 GPU.
- Inference must be fast enough for a real-time DJ app tagger.

### NFR-2: Continual Learning Scalability
- Adding new subgenres via user feedback must be near-instantaneous (computing a few MERT embeddings) without requiring full backpropagation.

---

## Out of Scope

- Full MAM pre-training from scratch on 8000 hours (we will use LoRA fine-tuning instead).
- Classifying non-EDM music.
