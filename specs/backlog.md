# Backlog

This document tracks future experiments, model improvements, and advanced features prioritized for post-MVP phases.

## Model Fine-Tuning & Architecture
- **Self-Supervised Fine-Tuning of MERT:** Train the MERT encoder further on a large, unlabeled corpus of electronic dance music (e.g., MTG-Jamendo) so it develops a deeper understanding of EDM acoustic structures.
- **Supervised Fine-Tuning of MERT:** Once we have gathered enough high-quality user-validated subgenre labels in our database, fine-tune the MERT classifier head for production-grade subgenre prediction accuracy.
- **Qwen2-Audio Experiments:** Prototype and evaluate Qwen2-Audio (or similar audio LLMs) as a conversational reasoning layer to handle ambiguous audio tracks or explain genre classifications to users.

## Advanced Learning Techniques
- **Reinforcement Learning with Human Feedback (RLHF):** Use user confirmations/corrections of subgenre predictions to continuously update and align the model.
- **Reinforcement Learning with Verifiable Rewards (RLVR):** Apply RLVR to improve the logical consistency and accuracy of the reasoning component.
