# Validation Plan: Hierarchical EDM Classifier

> References: [requirements.md](requirements.md), [plan.md](plan.md)

Each section maps to a functional or non-functional requirement and describes how to verify it is met.

---

## V1 — Hierarchical Multi-Label Prediction

**Target**: FR-1, FR-4

### Tests

| ID | What to verify | How |
|---|---|---|
| V1.1 | Multi-label support | Pass a track known to be a blend (e.g., Tech House) and assert both L1 House and L1 Techno pass the independent threshold. |
| V1.2 | DAG Constraint | Assert that when an L3 label (e.g., "Acid Techno") is predicted, its L2 ("Techno") and L1 parents are also predicted. |

---

## V2 — Continual Learning & Few-Shot Addition

**Target**: FR-3, NFR-2

### Tests

| ID | What to verify | How |
|---|---|---|
| V2.1 | Prototype Addition | Introduce 5 samples of a dummy new L3 genre ("Hyper-Trance"). Compute prototype. Ensure a 6th sample of "Hyper-Trance" is successfully classified without backprop. |
| V2.2 | No Catastrophic Forgetting | After adding a new prototype, verify accuracy on the base classes (House, Techno, etc.) remains strictly unchanged. |

---

## V3 — Compute Efficiency

**Target**: FR-2, NFR-1

### Tests

| ID | What to verify | How |
|---|---|---|
| V3.1 | LoRA Memory Usage | Monitor vRAM during training to ensure it stays well under the 24GB limit of an L4 GPU. |
| V3.2 | Temporal Aggregation Speed | Benchmark inference speed of chunking 30s into 5s segments vs. full 30s self-attention. |

---

## Definition of Done for Hierarchical Classifier

All of the following must be true before completing this feature:

- [ ] Unit tests pass for the DAG loss and contrastive head.
- [ ] Manual verification successful on an unseen test set of 100 DJ mixes.
- [ ] Code follows project standards.
