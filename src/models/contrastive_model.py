import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class RaveNet(nn.Module):
    def __init__(self, mert_model_id="m-a-p/MERT-v1-95M", projection_dim=256, chunk_length_sec=5.0, sample_rate=24000):
        super().__init__()
        self.chunk_length_sec = chunk_length_sec
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_length_sec * sample_rate)
        
        # Load base MERT encoder
        # MERT is based on Wav2Vec2 architecture
        print(f"Loading base MERT model from {mert_model_id}...")
        self.mert = AutoModel.from_pretrained(mert_model_id, trust_remote_code=True)
        
        # Extract config before PEFT wrap
        hidden_size = self.mert.config.hidden_size # usually 768
        
        # Apply LoRA to the attention layers
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"], # Wav2Vec2 self-attention usually has q_proj, k_proj, v_proj
            lora_dropout=0.05,
            bias="none",
        )
        self.mert = get_peft_model(self.mert, lora_config)
        print("LoRA applied to MERT encoder.")
        
        # We will use an attention pooling layer over the chunks
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, waveforms, attention_mask=None):
        """
        waveforms: (Batch, Total_Samples) e.g., 30 seconds of audio at 24000 Hz.
        attention_mask: (Batch, Total_Samples) boolean or 0/1 mask
        Returns: (Batch, projection_dim) normalized embedding
        """
        B, T = waveforms.shape
        
        # 1. Chunking
        # Pad if necessary to be a multiple of chunk_size
        if T % self.chunk_size != 0:
            pad_len = self.chunk_size - (T % self.chunk_size)
            waveforms = torch.nn.functional.pad(waveforms, (0, pad_len))
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
            T = waveforms.shape[1]
            
        num_chunks = T // self.chunk_size
        # Reshape to (Batch * num_chunks, chunk_size)
        chunks = waveforms.reshape(B * num_chunks, self.chunk_size)
        
        chunk_masks = None
        if attention_mask is not None:
            chunk_masks = attention_mask.reshape(B * num_chunks, self.chunk_size)
        
        # 2. Extract features with MERT
        # MERT expects input in shape (Batch, Sequence)
        # Returns hidden states of shape (B*num_chunks, seq_len, hidden_size)
        outputs = self.mert(input_values=chunks, attention_mask=chunk_masks)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling over the sequence dimension for each chunk
        # If chunk is all padding, its hidden states might be garbage, but we'll mask it out later
        chunk_embeddings = hidden_states.mean(dim=1)
        
        # 3. Temporal Aggregation over chunks
        # Reshape back to (Batch, num_chunks, hidden_size)
        chunk_embeddings = chunk_embeddings.reshape(B, num_chunks, -1)
        
        # Attention pooling
        # attn_weights: (Batch, num_chunks, 1)
        attn_logits = self.attention_pooling[:-1](chunk_embeddings) # up to the linear layer before softmax
        
        if chunk_masks is not None:
            # chunk_masks: (B*num_chunks, chunk_size)
            # Find chunks that are entirely padding (sum == 0)
            valid_chunks = chunk_masks.sum(dim=1) > 0 # (B*num_chunks,)
            valid_chunks = valid_chunks.reshape(B, num_chunks, 1)
            # Mask out invalid chunks with large negative number
            attn_logits = attn_logits.masked_fill(~valid_chunks, -1e9)
            
        attn_weights = self.attention_pooling[-1](attn_logits) # Softmax
        
        # clip_embedding: (Batch, hidden_size)
        clip_embedding = torch.sum(chunk_embeddings * attn_weights, dim=1)
        
        # 4. Projection
        # projected: (Batch, projection_dim)
        projected = self.projection_head(clip_embedding)
        
        # L2 Normalize for cosine similarity / contrastive loss
        projected = torch.nn.functional.normalize(projected, p=2, dim=1)
        
        return projected

    def register_prototype(self, label_idx: int, embeddings: torch.Tensor):
        """
        Compute and store the prototype for a given class from few-shot embeddings.
        embeddings: (N_shots, projection_dim)
        """
        if not hasattr(self, 'prototypes'):
            self.prototypes = {}
            
        prototype = embeddings.mean(dim=0)
        prototype = torch.nn.functional.normalize(prototype, p=2, dim=0)
        self.prototypes[label_idx] = prototype.detach()

    def predict(self, waveform: torch.Tensor, threshold: float = 0.5):
        """
        waveform: (1, Total_Samples)
        Returns: list of predicted label indices based on independent thresholding.
        """
        if not hasattr(self, 'prototypes') or len(self.prototypes) == 0:
            raise ValueError("No prototypes registered. Train or register prototypes first.")
            
        self.eval()
        with torch.no_grad():
            embedding = self.forward(waveform) # (1, projection_dim)
            
        predictions = []
        # Calculate cosine similarity with each prototype independently
        for label_idx, prototype in self.prototypes.items():
            # embedding: (1, dim), prototype: (dim)
            cos_sim = torch.sum(embedding[0] * prototype)
            # Use temperature scaled sigmoid or direct threshold on cosine similarity
            prob = torch.sigmoid(cos_sim * 10.0) 
            if prob.item() > threshold:
                predictions.append(label_idx)
                
        return predictions
