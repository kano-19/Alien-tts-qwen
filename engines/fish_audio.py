"""
Fish Audio S2 Pro Engine - Voice synthesis and cloning via Fish Audio API.

LICENSE NOTICE:
  Fish Audio S2 Pro is restricted to NON-COMMERCIAL use.
  License: Research / CC-NC
  You may use the code (MIT) but the MODEL is non-commercial only.
"""

import os
import io
import wave
import struct
import logging
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# License text shown in UI
LICENSE_WARNING = (
    "⚠️ AVISO DE LICENCIA: Fish Audio S2 Pro está bajo licencia "
    "Research / CC-NC. Solo para uso personal y no comercial. "
    "El uso comercial requiere licencia de Fish Audio."
)

# Check if fish-audio-sdk is available
try:
    from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
    FISH_AVAILABLE = True
except ImportError:
    FISH_AVAILABLE = False
    logger.warning("fish-audio-sdk not installed. Run: pip install fish-audio-sdk")


def is_available() -> bool:
    """Check if the Fish Audio SDK is installed."""
    return FISH_AVAILABLE


def get_api_key() -> str:
    """Get API key from environment or config file."""
    key = os.environ.get("FISH_API_KEY", "")
    if not key:
        config_path = os.path.join(os.path.dirname(__file__), "..", "fish_api_key.txt")
        config_path = os.path.normpath(config_path)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                key = f.read().strip()
    return key


def set_api_key(key: str) -> str:
    """Save API key to config file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "fish_api_key.txt")
    config_path = os.path.normpath(config_path)
    with open(config_path, "w") as f:
        f.write(key.strip())
    os.environ["FISH_API_KEY"] = key.strip()
    return "✅ API Key guardada correctamente."


def _pcm_bytes_to_numpy(audio_bytes: bytes, sample_rate: int = 44100) -> tuple[np.ndarray, int]:
    """Convert raw PCM/WAV bytes to numpy array."""
    try:
        # Try to read as WAV first
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                dtype = np.uint8

            audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
            if dtype == np.int16:
                audio /= 32768.0
            elif dtype == np.int32:
                audio /= 2147483648.0

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

            return audio, sr
    except Exception:
        pass

    # Fallback: treat as raw PCM int16
    n_samples = len(audio_bytes) // 2
    audio = np.array(
        struct.unpack(f"<{n_samples}h", audio_bytes[:n_samples * 2]),
        dtype=np.float32
    ) / 32768.0
    return audio, sample_rate


def generate(
    text: str,
    language: str = "Auto",
    ref_audio_path: Optional[str] = None,
    ref_text: str = "",
    model_id: str = "",
) -> tuple[np.ndarray, int]:
    """
    Generate speech using Fish Audio S2 Pro API.

    Args:
        text: Text to synthesize
        language: Target language
        ref_audio_path: Path to reference audio file for voice cloning
        ref_text: Transcript of reference audio
        model_id: Fish Audio model/voice ID (from fish.audio platform)

    Returns:
        Tuple of (wav_array, sample_rate)
    """
    if not FISH_AVAILABLE:
        raise RuntimeError(
            "fish-audio-sdk no está instalado.\n"
            "Ejecuta: pip install fish-audio-sdk"
        )

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "API Key de Fish Audio no configurada.\n"
            "1. Crea una cuenta en https://fish.audio\n"
            "2. Obtén tu API key del dashboard\n"
            "3. Pégala en la pestaña de Settings de Alien TTS"
        )

    session = Session(api_key)

    # Build request
    kwargs = {"text": text}

    # Voice cloning via reference audio
    if ref_audio_path:
        with open(ref_audio_path, "rb") as f:
            audio_bytes = f.read()
        ref = ReferenceAudio(
            audio=audio_bytes,
            text=ref_text if ref_text.strip() else None,
        )
        kwargs["references"] = [ref]

    # Use a specific model/voice from Fish Audio platform
    if model_id and model_id.strip():
        kwargs["reference_id"] = model_id.strip()

    # Generate
    logger.info(f"Generating with Fish Audio: {len(text)} chars, clone={bool(ref_audio_path)}")
    request = TTSRequest(**kwargs)

    # Collect audio chunks
    audio_data = bytearray()
    for chunk in session.tts(request):
        audio_data.extend(chunk)

    if not audio_data:
        raise RuntimeError("Fish Audio devolvió audio vacío.")

    # Convert to numpy
    wav, sr = _pcm_bytes_to_numpy(bytes(audio_data))
    logger.info(f"Fish Audio: generated {len(wav)/sr:.1f}s at {sr}Hz")
    return wav, sr
