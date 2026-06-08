import random
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, get_worker_info
import torchaudio

from src.audio.preprocess import preprocess_tensor, SAMPLE_RATE, CLIP_SECONDS


class RaveformStreamDataset(IterableDataset):
    """Streaming dataset that yields random 5-second segments from long-form DJ mixes.
    
    Scans the specified directory recursively for audio files and streams chunks dynamically
    to support large multi-gigabyte/terabyte datasets without loading them entirely into memory.
    """
    def __init__(
        self,
        data_dir: Path | str,
        chunks_per_file: int = 20,
        max_files: int | None = None,
        seed: int = 42
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.chunks_per_file = chunks_per_file
        self.seed = seed
        
        # Supported audio formats
        self.extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        
        # Gather all audio files recursively
        self.files = sorted([
            p for p in self.data_dir.rglob("*")
            if p.suffix.lower() in self.extensions
        ])
        
        if not self.files:
            raise ValueError(f"No audio files found in {data_dir}")
            
        if max_files is not None:
            random.seed(seed)
            self.files = random.sample(self.files, min(max_files, len(self.files)))
            
        print(f"RaveformStreamDataset initialized with {len(self.files)} mixes.")

    def __iter__(self):
        # Handle multi-worker splitting
        worker_info = get_worker_info()
        if worker_info is not None:
            # Partition file list among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # Seed worker randomly based on base seed and worker ID
            random.seed(self.seed + worker_id)
            worker_files = [
                path for idx, path in enumerate(self.files)
                if idx % num_workers == worker_id
            ]
        else:
            random.seed(self.seed)
            worker_files = self.files.copy()

        # Shuffle the files assigned to this worker at the start of each epoch
        random.shuffle(worker_files)

        for path in worker_files:
            try:
                # Retrieve audio metadata without loading the file content
                info = torchaudio.info(str(path))
                sr = info.sample_rate
                total_frames = info.num_frames
                total_seconds = total_frames / sr
                
                # We need files to be at least 5 seconds long
                if total_seconds < CLIP_SECONDS:
                    continue

                # Generate random offset start times for this file
                # To prevent out-of-bounds, clamp max start time to (total_seconds - 5.0)
                max_start_sec = total_seconds - CLIP_SECONDS
                start_times = [
                    random.uniform(0, max_start_sec)
                    for _ in range(self.chunks_per_file)
                ]

                for start_sec in start_times:
                    frame_offset = int(start_sec * sr)
                    num_frames = int(CLIP_SECONDS * sr)
                    
                    # Read only the specific 5-second slice from disk
                    waveform, file_sr = torchaudio.load(
                        str(path),
                        frame_offset=frame_offset,
                        num_frames=num_frames
                    )
                    
                    # Apply standard preprocessing (downsample to 24 kHz, stereo->mono, peak normalise)
                    preprocessed = preprocess_tensor(waveform, file_sr)
                    
                    # Yield preprocessed chunk
                    yield preprocessed
                    
            except Exception as e:
                # Log error and continue to the next mix to avoid training halts
                print(f"Warning: Failed to load/preprocess chunk from {path.name}: {e}")
                continue
