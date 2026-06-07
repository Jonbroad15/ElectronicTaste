import argparse
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.teachers import EnCodecTeacher, CQTTeacher
from src.models.mam_model import MERTWithMAMHeads, compute_mask_indices, get_device

# Set float32 matrix multiplication precision to high/medium for speedups on newer GPUs
torch.set_float32_matmul_precision("high")


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create a learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        # Cosine decay down to 10% of the peak learning rate
        return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(
        description="MERT Masked Audio Modeling (MAM) pre-training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True, help="Path to DJ mixes dataset directory")
    parser.add_argument("--checkpoint-dir", default="/mnt/data/models/mam_pretrain", help="Directory to save checkpoints")
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M", help="Base MERT model path")
    parser.add_argument("--steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=3000, help="Linear warmup steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--mask-prob", type=float, default=0.5, help="Time masking probability")
    parser.add_argument("--mask-length", type=int, default=10, help="Contiguous span length to mask")
    parser.add_argument("--lambda-cqt", type=float, default=10.0, help="Scaling factor for CQT reconstruction loss")
    parser.add_argument("--save-interval", type=int, default=5000, help="Steps between checkpoint saves")
    parser.add_argument("--device", default=None, help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Create directories
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Set device and seed
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {device}")
    print(f"Loading teachers...")
    # Initialize acoustic and musical target extractors (frozen)
    encodec_teacher = EnCodecTeacher(device=device)
    cqt_teacher = CQTTeacher(device=device)

    print(f"Loading MERT Model with MAM Heads...")
    # Initialize model to be pre-trained
    model = MERTWithMAMHeads(model_id=args.mert_model)
    model.to(device)
    model.train()  # Ensure model is in training mode for grads/dropout

    # Initialize optimizer (only train the MERT encoder + prediction heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Initialize dataset
    # We sample a random 5-second crop from each 30s chunk
    from src.training.dataset import ProcessedChunkDataset
    train_dir = Path(args.data_dir) / "train"
    dataset = ProcessedChunkDataset(
        processed_dir=train_dir,
        splits_file="splits.json",
        split_name="train",
        crop_seconds=5.0
    )
    # We only care about waveforms for MAM
    def collate_fn(batch):
        waveforms = torch.stack([item[0] for item in batch])
        return waveforms

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    print("Detecting model sequence length...")
    with torch.no_grad():
        dummy_input = torch.zeros(1, 120000, device=device)
        dummy_out = model.mert(dummy_input)
        seq_len = dummy_out.last_hidden_state.shape[1]
    print(f"MERT sequence length for 5s audio chunk is: {seq_len} frames")

    print("Starting pre-training loop...")
    step = 0
    t0 = time.time()
    
    # Iterate over stream infinite-loop style until total steps reached
    while step < args.steps:
        for waveforms in dataloader:
            if step >= args.steps:
                break
                
            # 1. Move raw audio waveforms to device
            waveforms = waveforms.to(device)  # Shape: (Batch, 120000)
            
            # 2. Extract target codes and CQT values from teachers
            # acoustic_targets shape: (Batch, 8, seq_len)
            # musical_targets shape: (Batch, 84, seq_len)
            with torch.no_grad():
                acoustic_targets = encodec_teacher.get_acoustic_codes(waveforms).to(device)
                musical_targets = cqt_teacher.get_cqt(waveforms).to(device)
                
            # Align teacher target sequence lengths to match MERT outputs
            
            # 3. Generate span mask
            # mask shape: (Batch, seq_len)
            mask = compute_mask_indices(
                shape=(waveforms.shape[0], seq_len),
                mask_prob=args.mask_prob,
                mask_length=args.mask_length,
                device=device
            )

            # 4. Mixed-precision forward pass
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # logits: (Batch, 8, seq_len, 1024), cqt: (Batch, seq_len, 84)
                acoustic_logits, cqt_logits = model(waveforms, mask_time_indices=mask)
                
                min_len = min(seq_len, acoustic_targets.shape[-1], musical_targets.shape[-1])
                acoustic_targets = acoustic_targets[:, :, :min_len]
                musical_targets = musical_targets[:, :, :min_len]
                acoustic_logits = acoustic_logits[:, :, :min_len, :]
                cqt_logits = cqt_logits[:, :min_len, :]
                mask_min = mask[:, :min_len]
                
                # Compute acoustic cross-entropy loss over the 8 codebooks ONLY on masked positions
                acoustic_loss = 0.0
                for c in range(8):
                    logits_c = acoustic_logits[:, c, :, :]  # (B, min_len, 1024)
                    targets_c = acoustic_targets[:, c, :]    # (B, min_len)
                    
                    # select only masked frames
                    masked_logits = logits_c[mask_min]           # (num_masked_frames, 1024)
                    masked_targets = targets_c[mask_min]         # (num_masked_frames,)
                    
                    loss_c = F.cross_entropy(masked_logits, masked_targets)
                    acoustic_loss += loss_c
                acoustic_loss = acoustic_loss / 8.0
                
                # Compute CQT reconstruction loss ONLY on masked positions
                targets_cqt = musical_targets.transpose(1, 2)  # (B, min_len, 84)
                masked_cqt_logits = cqt_logits[mask_min]           # (num_masked_frames, 84)
                masked_cqt_targets = targets_cqt[mask_min]         # (num_masked_frames, 84)
                
                loss_cqt = F.mse_loss(masked_cqt_logits, masked_cqt_targets)
                
                # Combined Loss
                loss = acoustic_loss + args.lambda_cqt * loss_cqt

            # 5. Optimize parameters
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            step += 1
            
            # 6. Logging
            if step % 50 == 0 or step == 1:
                elapsed = time.time() - t0
                steps_per_sec = step / max(0.1, elapsed)
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {step:6d}/{args.steps:6d} | "
                    f"Loss: {loss.item():.4f} (Acoustic: {acoustic_loss.item():.4f}, CQT: {loss_cqt.item():.4f}) | "
                    f"LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} steps/s"
                )

            # 7. Checkpointing
            if step % args.save_interval == 0 or step == args.steps:
                print(f"=== Step {step} — Saving Checkpoints ===")
                # Save full training state
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step,
                    "loss": loss.item(),
                }, ckpt_dir / f"mam_checkpoint_{step:06d}.pt")
                
                # Save MERT model adapter in Hugging Face format (so train.py can load it directly)
                adapted_mert_path = ckpt_dir / "mert_adapted"
                model.mert.save_pretrained(adapted_mert_path)
                print(f"  Saved adapted MERT encoder -> {adapted_mert_path}")

    print("Pre-training completed successfully!")


if __name__ == "__main__":
    main()
