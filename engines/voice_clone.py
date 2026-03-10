"""
Voice Clone Engine - Clone voices from reference audio.
Uses Qwen3-TTS Base models (1.7B / 0.6B).
"""

import logging
from typing import Optional, Union

import numpy as np

from model_manager import ModelManager

logger = logging.getLogger(__name__)

DEFAULT_KEY_1_7B = "voice_clone_1.7b"
DEFAULT_KEY_0_6B = "voice_clone_0.6b"


def get_model_key(model_size: str = "1.7B") -> str:
    """Get model key based on size preference."""
    if model_size == "0.6B":
        return DEFAULT_KEY_0_6B
    return DEFAULT_KEY_1_7B


def generate(
    text: str,
    language: str = "Auto",
    ref_audio: Union[str, tuple] = "",
    ref_text: str = "",
    x_vector_only_mode: bool = False,
    model_size: str = "1.7B",
) -> tuple[np.ndarray, int]:
    """
    Clone a voice and synthesize new text.

    Args:
        text: Text to synthesize with the cloned voice
        language: Target language
        ref_audio: Reference audio (file path, URL, or (numpy_array, sample_rate) tuple)
        ref_text: Transcript of the reference audio
        x_vector_only_mode: If True, only speaker embedding is used (no ref_text needed)
        model_size: "1.7B" or "0.6B"

    Returns:
        Tuple of (wav_array, sample_rate)
    """
    mgr = ModelManager()
    model_key = get_model_key(model_size)
    model = mgr.load(model_key)

    try:
        kwargs = {
            "text": text,
            "language": language,
            "ref_audio": ref_audio,
        }
        if ref_text and not x_vector_only_mode:
            kwargs["ref_text"] = ref_text
        if x_vector_only_mode:
            kwargs["x_vector_only_mode"] = True

        wavs, sr = model.generate_voice_clone(**kwargs)
        return wavs[0], sr
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(model_key)


def create_reusable_prompt(
    ref_audio: Union[str, tuple],
    ref_text: str = "",
    x_vector_only_mode: bool = False,
    model_size: str = "1.7B",
):
    """
    Create a reusable voice clone prompt for multiple generations.
    Avoids recomputing prompt features for each call.

    Args:
        ref_audio: Reference audio
        ref_text: Transcript of reference audio
        x_vector_only_mode: Use only speaker embedding
        model_size: "1.7B" or "0.6B"

    Returns:
        Reusable prompt object for generate_with_prompt()
    """
    mgr = ModelManager()
    model_key = get_model_key(model_size)
    model = mgr.load(model_key)

    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )
    return prompt


def generate_with_prompt(
    text: str,
    language: str = "Auto",
    voice_clone_prompt=None,
    model_size: str = "1.7B",
) -> tuple[np.ndarray, int]:
    """
    Generate speech using a pre-computed voice clone prompt.

    Args:
        text: Text to synthesize
        language: Target language
        voice_clone_prompt: Pre-computed prompt from create_reusable_prompt()
        model_size: "1.7B" or "0.6B"

    Returns:
        Tuple of (wav_array, sample_rate)
    """
    mgr = ModelManager()
    model_key = get_model_key(model_size)
    model = mgr.load(model_key)

    try:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )
        return wavs[0], sr
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(model_key)


def generate_batch_with_prompt(
    texts: list[str],
    languages: list[str],
    voice_clone_prompt=None,
    model_size: str = "1.7B",
) -> tuple[list[np.ndarray], int]:
    """
    Generate speech for multiple texts using a reusable prompt.

    Returns:
        Tuple of (list_of_wav_arrays, sample_rate)
    """
    mgr = ModelManager()
    model_key = get_model_key(model_size)
    model = mgr.load(model_key)

    try:
        wavs, sr = model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=voice_clone_prompt,
        )
        return wavs, sr
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(model_key)
