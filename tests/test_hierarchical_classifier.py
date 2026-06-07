import torch
import pytest
import time
from src.models.contrastive_model import RaveNet
from src.training.loss_functions import MultiLabelSupConLoss, DAGAwarePenalty

@pytest.fixture
def dummy_waveforms():
    # Batch size 2, 6 seconds of audio at 24000Hz
    return torch.randn(2, 6 * 24000)

def test_ravenet_lora_and_forward(dummy_waveforms):
    # Test V3.1 LoRA Memory/Params and forward pass
    model = RaveNet(chunk_length_sec=5.0)
    
    # Verify LoRA is active by checking trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    assert 0 < trainable_params < total_params * 0.1, "LoRA should dramatically reduce trainable parameters, but must have SOME trainable parameters!"
    
    # Test V3.2 temporal aggregation speed (rough check that forward works)
    t0 = time.time()
    embeddings = model(dummy_waveforms)
    t1 = time.time()
    
    assert embeddings.shape == (2, 256), "Output shape should be (Batch, projection_dim)"
    # Normalization check
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms)), "Embeddings must be L2 normalized"

def test_dag_aware_penalty():
    # Test V1.2 DAG Constraint
    penalty_fn = DAGAwarePenalty(margin=0.1)
    
    edges = [(0, 1), (1, 2)]
    
    # Scenario: violating DAG heavily
    probs_violating = torch.tensor([[0.1, 0.1, 0.9]])
    penalty = penalty_fn(probs_violating, edges)
    assert penalty.item() > 0, "Penalty should be applied for heavy DAG violation"
    
    # Scenario: marginal violation within margin
    probs_marginal = torch.tensor([[0.85, 0.85, 0.90]])
    penalty_marginal = penalty_fn(probs_marginal, edges)
    assert penalty_marginal.item() == 0.0, "No penalty if child exceeds parent by less than margin"

    # Scenario: small violation exceeding margin
    probs_small = torch.tensor([[0.75, 0.75, 0.90]])
    penalty_small = penalty_fn(probs_small, edges)
    assert penalty_small.item() > 0, "Penalty should apply if child exceeds parent by more than margin"

def test_prototype_addition_and_multilabel():
    # Test V2.1, V2.2, V1.1
    model = RaveNet()
    
    # Use random vectors (realistic) 
    house_embs = torch.randn(5, 256)
    techno_embs = torch.randn(5, 256)
    
    model.register_prototype(0, house_embs)
    model.register_prototype(1, techno_embs)
    
    house_proto_before = model.prototypes[0].clone()
    
    # Test V2.1: Add new subgenre (class 2 "Hyper-Trance")
    hyper_embs = torch.randn(3, 256)
    model.register_prototype(2, hyper_embs)
    
    assert len(model.prototypes) == 3
    # Test V2.2: Catastrophic forgetting check
    assert torch.equal(model.prototypes[0], house_proto_before), "Old prototypes must remain unchanged!"
    
    # Test V1.1: Multi-label support
    blend_emb = (model.prototypes[0] + model.prototypes[1]) / 2
    blend_emb = torch.nn.functional.normalize(blend_emb, p=2, dim=0)
    
    model.forward = lambda x: blend_emb.unsqueeze(0)
    
    # Predict with threshold 0.8 (since we added * 10 scaling in sigmoid, cos_sim > 0.13 triggers it)
    predictions = model.predict(torch.randn(1, 100), threshold=0.8)
    
    assert 0 in predictions, "Should predict L1 House"
    assert 1 in predictions, "Should predict L1 Techno"
    # Given 256 dims, random vector (Hyper-Trance) should have cos_sim near 0, so sigmoid(cos_sim*10) ~ 0.5 < 0.8
    assert 2 not in predictions, "Should not predict Hyper-Trance for a House/Techno blend"

def test_supcon_loss():
    loss_fn = MultiLabelSupConLoss(temperature=0.07)
    
    # Random embeddings with shared labels
    embeddings = torch.nn.functional.normalize(torch.randn(4, 256), p=2, dim=1)
    labels = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    
    loss_random = loss_fn(embeddings, labels)
    assert loss_random.item() > 0, "Loss should be computed successfully"
    
    # Perfect clusters should have lower loss
    perfect_embeddings = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    perfect_embeddings = torch.nn.functional.pad(perfect_embeddings, (0, 253))
    loss_perfect = loss_fn(perfect_embeddings, labels)
    assert loss_perfect.item() < loss_random.item(), "Perfectly clustered embeddings should have lower loss"
    
    # Edge case: No shared labels (silent zero bug check)
    isolated_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    isolated_embs = torch.randn(2, 256)
    loss_isolated = loss_fn(isolated_embs, isolated_labels)
    assert loss_isolated.item() == 0.0, "Should return 0.0 without NaNs when no positives exist in batch"
