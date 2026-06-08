import torch
from src.models.teachers import EnCodecTeacher, CQTTeacher

device = "cpu"
waveforms = torch.zeros(2, 120000)

encodec_teacher = EnCodecTeacher(device=device)
acoustic_targets = encodec_teacher.get_acoustic_codes(waveforms)

cqt_teacher = CQTTeacher(device=device)
musical_targets = cqt_teacher.get_cqt(waveforms)

print("Acoustic:", acoustic_targets.shape)
print("Musical:", musical_targets.shape)
