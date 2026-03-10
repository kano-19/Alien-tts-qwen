"""
Model Manager - Singleton for lazy loading/unloading Qwen3-TTS models.
Adapts behavior based on VRAM availability from system_check.
"""

import gc
import logging
from typing import Optional

import torch

from system_check import get_system_info, get_operating_mode

logger = logging.getLogger(__name__)

# Model registry: key -> (HuggingFace model ID, model_type)
MODEL_REGISTRY = {
    "custom_voice_1.7b": ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "custom_voice"),
    "custom_voice_0.6b": ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "custom_voice"),
    "voice_design_1.7b": ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "voice_design"),
    "voice_clone_1.7b":  ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "voice_clone"),
    "voice_clone_0.6b":  ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "voice_clone"),
    "tokenizer":         ("Qwen/Qwen3-TTS-Tokenizer-12Hz", "tokenizer"),
}


class ModelManager:
    """Manages lazy loading and unloading of Qwen3-TTS models."""

    _instance: Optional["ModelManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._loaded_models: dict = {}
        self._system_info = get_system_info()
        self._operating_mode = self._system_info["operating_mode"]
        self._check_flash_attn()
        logger.info(f"ModelManager initialized | mode={self._operating_mode}")

    def _check_flash_attn(self):
        """Check if FlashAttention 2 is available."""
        self._has_flash_attn = False
        try:
            import flash_attn  # noqa: F401
            self._has_flash_attn = True
            logger.info("FlashAttention 2 is available")
        except ImportError:
            logger.info("FlashAttention 2 not available, using default attention")

    @property
    def operating_mode(self) -> str:
        return self._operating_mode

    @property
    def system_info(self) -> dict:
        return self._system_info

    def _get_attn_impl(self) -> str:
        """Get the best attention implementation available."""
        if self._has_flash_attn:
            return "flash_attention_2"
        return "sdpa"

    def _get_device_map(self) -> str:
        """Get the device map based on GPU availability."""
        if self._system_info["gpu_available"]:
            return "cuda:0"
        return "cpu"

    def _get_dtype(self):
        """Get the optimal dtype for model loading."""
        if self._system_info["gpu_available"]:
            return torch.bfloat16
        return torch.float32

    def load(self, model_key: str):
        """
        Load a model by key. If already loaded, returns cached instance.
        In low-VRAM mode, unloads other models first.
        """
        if model_key in self._loaded_models:
            logger.info(f"Model '{model_key}' already loaded, returning cached")
            return self._loaded_models[model_key]

        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

        # In low VRAM mode, unload everything else first
        if self._operating_mode == "low":
            self.unload_all()
        elif self._operating_mode == "medium":
            # Keep at most 1 model loaded
            if len(self._loaded_models) >= 1:
                self.unload_all()

        model_id, model_type = MODEL_REGISTRY[model_key]
        logger.info(f"Loading model '{model_key}' ({model_id})...")

        if model_type == "tokenizer":
            from qwen_tts import Qwen3TTSTokenizer
            model = Qwen3TTSTokenizer.from_pretrained(
                model_id,
                device_map=self._get_device_map(),
            )
        else:
            from qwen_tts import Qwen3TTSModel
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self._get_device_map(),
                dtype=self._get_dtype(),
                attn_implementation=self._get_attn_impl(),
            )

        self._loaded_models[model_key] = model
        logger.info(f"Model '{model_key}' loaded successfully")
        return model

    def get(self, model_key: str):
        """Get a loaded model or load it if not cached."""
        return self.load(model_key)

    def unload(self, model_key: str):
        """Unload a specific model and free GPU memory."""
        if model_key in self._loaded_models:
            logger.info(f"Unloading model '{model_key}'...")
            del self._loaded_models[model_key]
            self._clear_gpu_memory()

    def unload_all(self):
        """Unload all models and free GPU memory."""
        if self._loaded_models:
            keys = list(self._loaded_models.keys())
            logger.info(f"Unloading all models: {keys}")
            self._loaded_models.clear()
            self._clear_gpu_memory()

    def _clear_gpu_memory(self):
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def is_loaded(self, model_key: str) -> bool:
        """Check if a model is currently loaded."""
        return model_key in self._loaded_models

    def get_loaded_models(self) -> list:
        """Get list of currently loaded model keys."""
        return list(self._loaded_models.keys())

    def get_vram_usage(self) -> dict:
        """Get current VRAM usage stats."""
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        }

    def should_unload_after_use(self) -> bool:
        """Whether models should be unloaded immediately after inference."""
        return self._operating_mode in ("low", "cpu")
