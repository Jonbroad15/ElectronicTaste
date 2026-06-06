"""AudioDataset and LabelEncoder for classifier training.

Expected directory layout::

    root/
        blues/
            blues.00000.wav
            blues.00001.wav
        classical/
            ...

The dataset caches preprocessed waveform tensors to ``cache_dir`` so that
repeated training runs skip the resampling step.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.audio.preprocess import load_and_preprocess, SAMPLE_RATE, CLIP_SECONDS

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


# ── Label encoder ────────────────────────────────────────────────────────────

class LabelEncoder:
    """Bidirectional mapping between string label names and integer indices.

    Args:
        names: Ordered list of class names.  Index in the list equals the
               integer label used by the model.
    """

    def __init__(self, names: list[str] | None = None) -> None:
        self._names: list[str] = names or []
        self._index: dict[str, int] = {n: i for i, n in enumerate(self._names)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Ordered list of class names."""
        return self._names

    def __len__(self) -> int:
        return len(self._names)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, label: str) -> int:
        """Map a label name to its integer index.

        Raises:
            KeyError: if ``label`` is not in the encoder.
        """
        return self._index[label]

    def decode(self, idx: int) -> str:
        """Map an integer index back to a label name."""
        return self._names[idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save label names to a JSON file."""
        Path(path).write_text(json.dumps(self._names, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LabelEncoder":
        """Load a LabelEncoder from a JSON file written by :meth:`save`."""
        names = json.loads(Path(path).read_text())
        return cls(names)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(cls, root: str | Path) -> "LabelEncoder":
        """Build an encoder from sorted subdirectory names under ``root``.

        Each subdirectory is treated as one class.  Sorting ensures
        deterministic ordering across runs.
        """
        root = Path(root)
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
        return cls(names)


# ── Dataset ──────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """PyTorch dataset that loads and preprocesses audio files.

    Args:
        root:          Root directory with per-class subdirectories.
        label_encoder: :class:`LabelEncoder` whose names match the subdir
                       names under ``root``.
        cache_dir:     Optional directory for caching preprocessed tensors.
                       Pass ``None`` to disable caching.
        clip_seconds:  Length of each clip window (default 5 s).
        target_sr:     Target sample rate (default 24 kHz for MERT).
    """

    def __init__(
        self,
        root: str | Path,
        label_encoder: LabelEncoder,
        cache_dir: str | Path | None = None,
        clip_seconds: float = CLIP_SECONDS,
        target_sr: int = SAMPLE_RATE,
    ) -> None:
        self.root = Path(root)
        self.label_encoder = label_encoder
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.clip_seconds = clip_seconds
        self.target_sr = target_sr
        self.samples: list[tuple[Path, int]] = []
        self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        """Walk ``root`` and collect ``(audio_path, label_index)`` pairs."""
        for label_dir in sorted(self.root.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label_idx = self.label_encoder.encode(label_dir.name)
            except KeyError:
                continue  # skip subdirs not in the encoder
            for audio_file in sorted(label_dir.glob("*")):
                if audio_file.suffix.lower() in _AUDIO_EXTENSIONS:
                    self.samples.append((audio_file, label_idx))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        waveform = self._load_cached(path)
        return waveform, label

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_key(self, audio_path: Path) -> Path | None:
        if self.cache_dir is None:
            return None
        # Include preprocessing parameters in the key so that changing
        # clip_seconds or target_sr invalidates stale cached tensors.
        param_tag = f"sr{self.target_sr}_clip{self.clip_seconds:.1f}"
        safe = audio_path.as_posix().replace("/", "__").replace(".", "_")
        return self.cache_dir / f"{safe}__{param_tag}.pkl"

    def _load_cached(self, path: Path) -> torch.Tensor:
        cache_path = self._cache_key(path)
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

        waveform = load_and_preprocess(
            str(path),
            target_sr=self.target_sr,
            clip_seconds=self.clip_seconds,
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(waveform, fh)

        return waveform
