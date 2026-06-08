import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: (Batch, projection_dim) - L2 normalized embeddings
        labels: (Batch, Num_Classes) - Multi-hot encoded labels (0s and 1s)
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute cosine similarity matrix
        # Since embeddings are L2 normalized, dot product is cosine sim
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(mask, -9e15)

        # Create positive weights: using Intersection-over-Union (IoU) of labels
        shared_labels = torch.matmul(labels, labels.T)
        label_sums = labels.sum(dim=1, keepdim=True)
        union_labels = label_sums + label_sums.T - shared_labels
        positive_weights = shared_labels / (union_labels + 1e-8)
        positive_weights.masked_fill_(mask, 0.0) # remove self

        # For numerical stability
        max_sim, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        sim_stable = similarity_matrix - max_sim.detach()

        # Compute log probabilities
        exp_sim = torch.exp(sim_stable)
        log_prob = sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood weighted by positive IoU weights
        mean_log_prob_pos = (positive_weights * log_prob).sum(dim=1) / (positive_weights.sum(dim=1) + 1e-8)

        # Loss (avoid silent zero averaging bug)
        valid_mask = positive_weights.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = -mean_log_prob_pos[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss


class DAGAwarePenalty(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, probabilities, hierarchy_edges):
        """
        probabilities: (Batch, Num_Classes) - predicted probabilities (e.g., via Sigmoid(cosine_sim))
        hierarchy_edges: list of tuples (parent_idx, child_idx)
        """
        if not hierarchy_edges:
            return torch.tensor(0.0, device=probabilities.device, requires_grad=True)
            
        parents = [p for p, c in hierarchy_edges]
        children = [c for p, c in hierarchy_edges]
        
        parent_probs = probabilities[:, parents]
        child_probs = probabilities[:, children]
        
        # soft margin penalty
        violations = F.relu(child_probs - parent_probs - self.margin)
        penalty = violations.mean()
            
        return penalty
