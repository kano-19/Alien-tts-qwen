"""
Audio Upscaler - Optional 24kHz to 48kHz upscaling using high-quality resampling.
Uses scipy polyphase resampling with optional spectral enhancement.
"""

import logging

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


def upscale(
    wav: np.ndarray,
    sr_in: int = 24000,
    sr_out: int = 48000,
    enhance: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Upscale audio from sr_in to sr_out using high-quality polyphase resampling.

    Args:
        wav: Input waveform as numpy array
        sr_in: Input sample rate (default 24000 for Qwen3-TTS)
        sr_out: Target sample rate (default 48000)
        enhance: Apply spectral enhancement for better quality

    Returns:
        Tuple of (upscaled_wav, sr_out)
    """
    if sr_in == sr_out:
        return wav, sr_out

    # Calculate resampling ratio
    from math import gcd
    g = gcd(sr_out, sr_in)
    up = sr_out // g
    down = sr_in // g

    logger.info(f"Upscaling audio: {sr_in}Hz → {sr_out}Hz (ratio {up}/{down})")

    # High-quality polyphase resampling
    upscaled = resample_poly(wav, up, down)

    if enhance:
        upscaled = _spectral_enhance(upscaled, sr_out)

    # Normalize to prevent clipping
    peak = np.max(np.abs(upscaled))
    if peak > 0:
        upscaled = upscaled * (0.95 / peak)

    return upscaled.astype(np.float32), sr_out


def _spectral_enhance(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply subtle spectral enhancement to add presence to upscaled audio.
    Boosts high-frequency content slightly for more clarity.
    """
    try:
        from scipy.signal import butter, filtfilt

        # Gentle high-shelf boost: emphasize the 4kHz-12kHz range
        # that was reconstructed during upsampling
        nyq = sr / 2
        high_cutoff = min(12000, nyq * 0.9) / nyq

        # Design a gentle high-pass for the enhancement signal
        b, a = butter(2, high_cutoff * 0.3, btype='high')
        high_freq = filtfilt(b, a, wav)

        # Mix: original + subtle high-frequency boost
        enhanced = wav + high_freq * 0.08

        return enhanced.astype(np.float32)
    except Exception as e:
        logger.warning(f"Spectral enhancement failed, returning original: {e}")
        return wav


def get_info() -> str:
    """Get description of the upscaler for UI display."""
    return (
        "High-quality polyphase resampling (24kHz → 48kHz) "
        "with optional spectral enhancement for added clarity. "
        "No additional model downloads required."
    )
