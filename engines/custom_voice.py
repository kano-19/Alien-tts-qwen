"""
Custom Voice Engine - Uses Qwen3-TTS CustomVoice models with built-in speakers.
Supports emotion/style control via instruct parameter.
"""

import logging
from typing import Optional

import numpy as np

from model_manager import ModelManager

logger = logging.getLogger(__name__)

# Default model key
DEFAULT_KEY_1_7B = "custom_voice_1.7b"
DEFAULT_KEY_0_6B = "custom_voice_0.6b"


def get_model_key(model_size: str = "1.7B") -> str:
    """Get model key based on size preference."""
    if model_size == "0.6B":
        return DEFAULT_KEY_0_6B
    return DEFAULT_KEY_1_7B


def get_speakers(model_size: str = "1.7B") -> list:
    """Get list of supported speakers for the CustomVoice model."""
    mgr = ModelManager()
    model = mgr.load(get_model_key(model_size))
    try:
        speakers = model.get_supported_speakers()
        return list(speakers) if speakers else []
    except Exception as e:
        logger.warning(f"Could not get speakers: {e}")
        return ["Vivian", "Ryan", "Chelsie", "Ethan", "Aria", "Benjamin"]
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(get_model_key(model_size))


def get_languages(model_size: str = "1.7B") -> list:
    """Get list of supported languages."""
    mgr = ModelManager()
    model = mgr.load(get_model_key(model_size))
    try:
        languages = model.get_supported_languages()
        return list(languages) if languages else []
    except Exception as e:
        logger.warning(f"Could not get languages: {e}")
        return [
            "Auto", "Chinese", "English", "Japanese", "Korean",
            "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
        ]
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(get_model_key(model_size))


def generate(
    text: str,
    language: str = "Auto",
    speaker: str = "Vivian",
    instruct: str = "",
    model_size: str = "1.7B",
) -> tuple[np.ndarray, int]:
    """
    Generate speech using a built-in custom voice.

    Args:
        text: Text to synthesize
        language: Target language (or "Auto")
        speaker: Speaker name from the model's supported list
        instruct: Emotion/style instruction (e.g., "speak angrily", "whispered")
        model_size: "1.7B" or "0.6B"

    Returns:
        Tuple of (wav_array, sample_rate)
    """
    mgr = ModelManager()
    model_key = get_model_key(model_size)
    model = mgr.load(model_key)

    try:
        # Qwen3-TTS expects capitalized language names (e.g., "Spanish" not "spanish")
        language = language.strip().title() if language else "Auto"

        kwargs = {
            "text": text,
            "language": language,
            "speaker": speaker,
        }
        if instruct and instruct.strip():
            kwargs["instruct"] = instruct

        wavs, sr = model.generate_custom_voice(**kwargs)
        return wavs[0], sr
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(model_key)


def generate_batch(
    texts: list[str],
    languages: list[str],
    speakers: list[str],
    instructs: Optional[list[str]] = None,
    model_size: str = "1.7B",
) -> tuple[list[np.ndarray], int]:
    """
    Generate speech for multiple texts in a batch.

    Returns:
        Tuple of (list_of_wav_arrays, sample_rate)
    """
    mgr = ModelManager()
    model_key = get_model_key(model_size)
    model = mgr.load(model_key)

    try:
        kwargs = {
            "text": texts,
            "language": languages,
            "speaker": speakers,
        }
        if instructs:
            kwargs["instruct"] = instructs

        wavs, sr = model.generate_custom_voice(**kwargs)
        return wavs, sr
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(model_key)
