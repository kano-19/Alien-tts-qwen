"""
Voice Design Engine - Create custom voices from natural language descriptions.
Uses Qwen3-TTS VoiceDesign model (1.7B only).
"""

import logging
from typing import Optional

import numpy as np

from model_manager import ModelManager

logger = logging.getLogger(__name__)

MODEL_KEY = "voice_design_1.7b"


def generate(
    text: str,
    language: str = "Auto",
    instruct: str = "",
) -> tuple[np.ndarray, int]:
    """
    Generate speech with a designed voice from natural language description.

    Args:
        text: Text to synthesize
        language: Target language
        instruct: Natural language voice description (e.g.,
                  "Female, 25 years old, warm and gentle voice with slight breathiness")

    Returns:
        Tuple of (wav_array, sample_rate)
    """
    mgr = ModelManager()
    model = mgr.load(MODEL_KEY)

    try:
        # Qwen3-TTS expects capitalized language names
        language = language.strip().title() if language else "Auto"

        wavs, sr = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
        return wavs[0], sr
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(MODEL_KEY)


def generate_batch(
    texts: list[str],
    languages: list[str],
    instructs: list[str],
) -> tuple[list[np.ndarray], int]:
    """
    Generate speech for multiple texts with designed voices.

    Returns:
        Tuple of (list_of_wav_arrays, sample_rate)
    """
    mgr = ModelManager()
    model = mgr.load(MODEL_KEY)

    try:
        wavs, sr = model.generate_voice_design(
            text=texts,
            language=languages,
            instruct=instructs,
        )
        return wavs, sr
    finally:
        if mgr.should_unload_after_use():
            mgr.unload(MODEL_KEY)


def design_then_clone_reference(
    ref_text: str,
    language: str = "English",
    instruct: str = "",
) -> tuple[np.ndarray, int]:
    """
    Design a voice and return the reference audio for cloning.
    This is step 1 of the "Design then Clone" workflow.

    Args:
        ref_text: A short sentence to generate as reference
        language: Language for the reference
        instruct: Voice description

    Returns:
        Tuple of (wav_array, sample_rate) to be used as clone reference
    """
    return generate(
        text=ref_text,
        language=language,
        instruct=instruct,
    )
