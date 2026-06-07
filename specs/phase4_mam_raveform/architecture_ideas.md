# Architecture Ideas: Hierarchical EDM Subgenre Classifier

This document outlines proposed machine learning architectures and training strategies for classifying 30-second EDM samples into L1, L2, and L3 hierarchical subgenres. It also considers continuous learning for a deployed application.

## 1. MERT Pretraining Strategy (MAM)

The dataset consists of ~8000 hours (nearly 1 million 30s samples) of EDM music. To specialize a model to EDM, we will utilize the 95M parameter MERT (Acoustic Music undERstanding model with large-scale self-supervised Training) backbone.

*   **Objective:** Masked Acoustic Modeling (MAM).
*   **Teachers:** 
    *   **Acoustic Teacher:** EnCodec or RVQ-VAE to provide discrete acoustic tokens.
    *   **Musical Teacher:** Constant-Q Transform (CQT) to capture pitch/harmonic bias, crucial for distinguishing subtle melodic subgenres.
*   **Adaptation:** Apply EDM-specific augmentations during pretraining, such as tempo stretching and pitch shifting, and in-batch noise mixup, to build robust representations.
*   **Viability & Compute Note:** Fully continuing MAM pretraining on 8000 hours requires significant compute clusters. Due to the small model size (95M params), MAM pre-training will utilize **full fine-tuning** (no LoRA) on a smaller subset/step count. Parameter-Efficient Fine-Tuning (PEFT), specifically **LoRA**, is reserved for the downstream RaveNet classifier to prevent catastrophic forgetting of the acoustic knowledge.

## 2. Hierarchical Multi-Label Classifier Architecture

Once the 95M MERT backbone is adapted to the EDM domain, we will build a classification head on top.

*   **Feature Extraction:** The pre-trained 95M MERT encoder outputs sequence embeddings. We will use **LoRA** (Low-Rank Adaptation) inserted into the attention layers to allow the embedding space to smoothly adapt without catastrophic forgetting.
*   **Temporal Aggregation:** Since 30s is a long context, standard self-attention has a high quadratic memory cost. We will chunk the 30s sample into smaller 5s segments, apply pooling to these segments, and then aggregate the segment embeddings to form the final clip-level representation.
*   **Hierarchical Prediction Heads:** 
    *   We use a Multi-Task Learning (MTL) approach with three parallel branches for L1, L2, and L3 classifications.
    *   **Hierarchical Constraints:** To enforce the ontology (e.g., L3 "Acid Techno" must have L2 "Techno" and L1 "Techno/Trance"), we will implement a DAG-aware loss function. 
    *   **Overfitting Risk:** To prevent the model from overfitting to dominant hierarchical paths and penalizing rare multi-label combinations, we will apply **hierarchical label smoothing** or a soft margin to the DAG loss.

## 3. Deployment and Continual Learning (Few-Shot)

The deployed app will allow users to provide feedback and introduce new subgenres. The model must adapt quickly without catastrophic forgetting or full retraining.

*   **Prototypical Contrastive Learning Space:** Instead of standard cross-entropy linear classification layers, we will train the projection layer using Supervised Contrastive Learning (SupCon). This groups samples of the same subgenre tightly in the embedding space.
*   **Nearest Centroid Classifier (Multi-Label Handling):** Standard softmax prototypical networks assume mutually exclusive classes. To handle multi-label samples:
    1.  **Independent Thresholding:** Calculate the cosine similarity between the sample embedding and *each* class prototype independently. Apply a Sigmoid and a calibrated threshold per class.
    2.  **Sub-clustering:** A single prototype might poorly represent a diverse multi-label class. We can use multiple prototypes per class (e.g., k-means clustering the class samples in embedding space) and match against the nearest sub-prototype.
*   **Few-Shot Class Incremental Learning (FSCIL):** When a user introduces a new subgenre with only a few samples, we pass them through the MERT encoder, compute the new prototype vector (or sub-prototypes), and insert it into the hierarchical graph. The model can instantly classify the new subgenre without any immediate backpropagation or retraining.
