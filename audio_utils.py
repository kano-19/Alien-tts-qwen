"""
Audio Utilities - Merge chunks, normalize, and export audio files.
"""

import os
import logging
from datetime import datetime

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def merge_chunks(
    chunks: list[np.ndarray],
    sr: int,
    pauses_ms: list[int] = None,
) -> np.ndarray:
    """
    Merge multiple audio chunks into a single waveform with pauses between them.

    Args:
        chunks: List of numpy audio arrays
        sr: Sample rate
        pauses_ms: List of pause durations in ms between chunks.
                   If None or shorter, uses 100ms default gaps.

    Returns:
        Merged audio as numpy array
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    if len(chunks) == 1:
        return chunks[0]

    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(chunk)

        # Add pause between chunks (not after the last one)
        if i < len(chunks) - 1:
            pause_ms = 100  # default gap
            if pauses_ms and i < len(pauses_ms):
                pause_ms = max(pauses_ms[i], 0)

            if pause_ms > 0:
                silence_samples = int(sr * pause_ms / 1000)
                parts.append(np.zeros(silence_samples, dtype=np.float32))

    return np.concatenate(parts)


def normalize(wav: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak-normalize audio to target level.

    Args:
        wav: Audio waveform
        target_peak: Target peak amplitude (0.0 - 1.0)

    Returns:
        Normalized audio
    """
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav * (target_peak / peak)
    return wav


def export(
    wav: np.ndarray,
    sr: int,
    path: str = None,
    format: str = "wav",
    normalize_audio: bool = True,
) -> str:
    """
    Export audio to a file.

    Args:
        wav: Audio waveform
        sr: Sample rate
        path: Output file path. If None, auto-generates in outputs/
        format: "wav" or "mp3"
        normalize_audio: Whether to normalize before saving

    Returns:
        Path to the saved file
    """
    if normalize_audio:
        wav = normalize(wav)

    if path is None:
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "wav" if format == "wav" else "mp3"
        path = os.path.join("outputs", f"tts_{timestamp}.{ext}")

    if format == "mp3":
        # Use pydub for MP3 export
        try:
            from pydub import AudioSegment
            import io

            # Convert to int16 for pydub
            wav_int16 = (wav * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                data=wav_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            audio_segment.export(path, format="mp3", bitrate="320k")
        except ImportError:
            # Fallback to WAV if pydub not available
            logger.warning("pydub not available, saving as WAV instead")
            path = path.rsplit(".", 1)[0] + ".wav"
            sf.write(path, wav, sr)
    else:
        sf.write(path, wav, sr)

    logger.info(f"Audio exported to: {path}")
    return path


def get_duration_str(wav: np.ndarray, sr: int) -> str:
    """Get a human-readable duration string."""
    seconds = len(wav) / sr
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def make_gradio_audio(wav: np.ndarray, sr: int) -> tuple:
    """Create a (sample_rate, waveform) tuple for Gradio audio output."""
    return (sr, wav)
