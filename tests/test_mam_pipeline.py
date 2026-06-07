import pytest
import torch
from src.models.teachers import EnCodecTeacher, CQTTeacher
from src.models.mam_model import MERTWithMAMHeads, compute_mask_indices
from src.training.mam_dataset import RaveformStreamDataset


def test_teachers():
    """Verify that EnCodec and CQT teachers process waveforms and return the correct shape."""
    # EnCodec and CQT teachers initialized on CPU
    encodec_teacher = EnCodecTeacher(device="cpu")
    cqt_teacher = CQTTeacher(device="cpu")

    # 1 second of audio at 24 kHz
    batch_size = 2
    samples = 24000
    mock_waveform = torch.randn(batch_size, samples)

    # Extract targets
    acoustic_codes = encodec_teacher.get_acoustic_codes(mock_waveform)
    cqt_spec = cqt_teacher.get_cqt(mock_waveform)

    # 24,000 samples / 320 hop_length = 75 frames
    expected_seq_len = 75

    assert acoustic_codes.shape == (batch_size, 8, expected_seq_len)
    assert cqt_spec.shape == (batch_size, 84, expected_seq_len)


def test_compute_mask_indices():
    """Verify mask indices generation helper."""
    batch_size = 3
    seq_len = 375
    mask_prob = 0.5
    mask_length = 10

    mask = compute_mask_indices(
        shape=(batch_size, seq_len),
        mask_prob=mask_prob,
        mask_length=mask_length,
        device=torch.device("cpu")
    )

    assert mask.shape == (batch_size, seq_len)
    assert mask.dtype == torch.bool
    # At least some frames should be masked
    assert mask.any()
    # Masking density should be approximately close to mask_prob (given random sample variance)
    masked_fraction = mask.float().mean().item()
    assert 0.25 < masked_fraction < 0.75


def test_mam_model_forward():
    """Verify MERT wrapper forward pass with custom mask indices."""
    device = torch.device("cpu")
    
    # Initialize wrapper
    model = MERTWithMAMHeads(model_id="m-a-p/MERT-v1-95M")
    model.to(device)
    model.eval()

    # 5 seconds of audio at 24 kHz
    batch_size = 2
    samples = 120000
    mock_waveform = torch.randn(batch_size, samples, device=device)

    # Determine sequence length dynamically from the model
    with torch.no_grad():
        dummy_out = model.mert(mock_waveform[:1])
        seq_len = dummy_out.last_hidden_state.shape[1]

    # Generate mask
    mask = compute_mask_indices(
        shape=(batch_size, seq_len),
        mask_prob=0.5,
        mask_length=10,
        device=device
    )

    # Forward pass
    with torch.no_grad():
        acoustic_logits, cqt_logits = model(mock_waveform, mask_time_indices=mask)

    assert acoustic_logits.shape == (batch_size, 8, seq_len, 1024)
    assert cqt_logits.shape == (batch_size, seq_len, 84)
