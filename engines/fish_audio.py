"""
Fish Audio Engine - Local inference using Fish Speech S1-mini.

This engine runs ENTIRELY locally. Models are downloaded on-demand
only when the user selects this mode for the first time.

LICENSE NOTICE:
  Fish Audio S2 Pro / S1 models are restricted to NON-COMMERCIAL use.
  License: Research / CC-NC
  You may use the code (MIT) but the MODEL is non-commercial only.

Supported models (auto-selected by VRAM):
  - fishaudio/s1-mini  (0.5B params, ~4GB VRAM)  ← default for <12GB
  - fishaudio/s2-pro   (larger, ~12GB VRAM)       ← if available
"""

import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
FISH_DIR = PROJECT_ROOT / "fish-speech"
CHECKPOINTS_DIR = FISH_DIR / "checkpoints"

LICENSE_WARNING = (
    "⚠️ AVISO DE LICENCIA: Fish Audio está bajo licencia "
    "Research / CC-NC. Solo para uso personal y no comercial. "
    "El uso comercial requiere licencia de Fish Audio."
)

# ── State ────────────────────────────────────────────────────────────────

_installed = None
_model_loaded = False
_model_variant = None  # "s1-mini" or "s2-pro"


def is_installed() -> bool:
    """Check if fish-speech is installed and ready."""
    global _installed
    if _installed is not None:
        return _installed
    try:
        import fish_speech
        _installed = True
    except ImportError:
        _installed = False
    return _installed


def get_available_models() -> list[str]:
    """Return list of downloaded model variants."""
    models = []
    if CHECKPOINTS_DIR.exists():
        for d in CHECKPOINTS_DIR.iterdir():
            if d.is_dir() and (d / "codec.pth").exists():
                models.append(d.name)
    return models


def get_recommended_model(vram_gb: float) -> str:
    """Pick best model for available VRAM."""
    if vram_gb >= 12:
        return "s2-pro"
    return "s1-mini"


def install_fish_speech(progress_callback=None) -> str:
    """
    Install fish-speech package locally by cloning the repo.
    Only runs once, skips if already installed.
    """
    global _installed

    if is_installed():
        return "✅ Fish Speech ya está instalado."

    if progress_callback:
        progress_callback(0.1, desc="🐟 Clonando fish-speech...")

    # Clone if not present
    if not FISH_DIR.exists():
        logger.info("Cloning fish-speech repository...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/fishaudio/fish-speech.git",
             str(FISH_DIR)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            return f"❌ Error clonando: {result.stderr}"

    if progress_callback:
        progress_callback(0.4, desc="🐟 Instalando dependencias de fish-speech...")

    # Install the package
    logger.info("Installing fish-speech package...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
        capture_output=True, text=True, cwd=str(FISH_DIR)
    )
    if result.returncode != 0:
        return f"❌ Error instalando: {result.stderr[:500]}"

    _installed = True

    if progress_callback:
        progress_callback(0.7, desc="✅ Fish Speech instalado!")

    return "✅ Fish Speech instalado correctamente."


def download_model(model_name: str = "s1-mini", progress_callback=None, hf_token: str = "") -> str:
    """
    Download model weights from HuggingFace.
    Only downloads if not already present.
    """
    checkpoint_dir = CHECKPOINTS_DIR / model_name

    if checkpoint_dir.exists() and (checkpoint_dir / "codec.pth").exists():
        return f"✅ Modelo {model_name} ya descargado."

    if progress_callback:
        progress_callback(0.3, desc=f"🐟 Descargando modelo {model_name}...")

    hf_repo = f"fishaudio/{model_name}"
    logger.info(f"Downloading model weights: {hf_repo}")

    # Set HF token environment variable if provided
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    result = subprocess.run(
        [sys.executable, "-m", "huggingface_hub.cli.main", "download",
         hf_repo, "--local-dir", str(checkpoint_dir)] + (["--token", hf_token] if hf_token else []),
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        # Try alternate download method
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(hf_repo, local_dir=str(checkpoint_dir), token=hf_token if hf_token else None)
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "gated repo" in err_msg.lower():
                return f"❌ Error: El modelo {hf_repo} requiere acceso. Pega tu HuggingFace Token (Read) en la interfaz."
            return f"❌ Error descargando modelo: {err_msg}"

    if progress_callback:
        progress_callback(0.8, desc=f"✅ Modelo {model_name} listo!")

    return f"✅ Modelo {model_name} descargado en {checkpoint_dir}"


def setup_if_needed(model_name: str = "s1-mini", progress_callback=None) -> str:
    """Install fish-speech and download model if not already done."""
    status_parts = []

    if not is_installed():
        msg = install_fish_speech(progress_callback)
        status_parts.append(msg)
        if "❌" in msg:
            return msg

    checkpoint_dir = CHECKPOINTS_DIR / model_name
    if not checkpoint_dir.exists() or not (checkpoint_dir / "codec.pth").exists():
        msg = download_model(model_name, progress_callback)
        status_parts.append(msg)
        if "❌" in msg:
            return msg

    return " | ".join(status_parts) if status_parts else "✅ Todo listo."


def generate(
    text: str,
    ref_audio_path: Optional[str] = None,
    ref_text: str = "",
    model_name: str = "s1-mini",
) -> tuple[np.ndarray, int]:
    """
    Generate speech using local Fish Speech inference.

    Pipeline:
      1. Encode reference audio → VQ tokens (if cloning)
      2. Generate semantic tokens from text
      3. Decode semantic tokens → audio

    Args:
        text: Text to synthesize
        ref_audio_path: Optional path to reference audio for voice cloning
        ref_text: Transcript of reference audio
        model_name: "s1-mini" or "s2-pro"

    Returns:
        Tuple of (wav_array, sample_rate)
    """
    if not is_installed():
        raise RuntimeError(
            "Fish Speech no está instalado. "
            "Selecciona el modo Fish Audio para instalarlo automáticamente."
        )

    checkpoint_dir = CHECKPOINTS_DIR / model_name
    if not checkpoint_dir.exists():
        raise RuntimeError(
            f"Modelo {model_name} no descargado. "
            "Se descargará automáticamente la primera vez."
        )

    # Use subprocess to run inference scripts from fish-speech
    # This avoids conflicts between the two PyTorch installations
    output_dir = tempfile.mkdtemp(prefix="fish_")
    output_wav = os.path.join(output_dir, "output.wav")

    try:
        # Step 1: Encode reference audio (if voice cloning)
        prompt_tokens_path = None
        if ref_audio_path:
            logger.info("Step 1: Encoding reference audio...")
            result = subprocess.run(
                [sys.executable,
                 str(FISH_DIR / "fish_speech" / "models" / "dac" / "inference.py"),
                 "-i", ref_audio_path,
                 "--checkpoint-path", str(checkpoint_dir / "codec.pth"),
                 ],
                capture_output=True, text=True, cwd=str(FISH_DIR)
            )
            if result.returncode != 0:
                logger.warning(f"Codec encoding warning: {result.stderr[:300]}")
            prompt_tokens_path = os.path.join(FISH_DIR, "fake.npy")

        # Step 2: Generate semantic tokens from text
        logger.info("Step 2: Generating semantic tokens...")
        cmd = [
            sys.executable,
            str(FISH_DIR / "fish_speech" / "models" / "text2semantic" / "inference.py"),
            "--text", text,
            "--half",  # Use FP16 for lower VRAM
        ]
        if prompt_tokens_path and os.path.exists(prompt_tokens_path):
            cmd.extend(["--prompt-tokens", prompt_tokens_path])
        if ref_text:
            cmd.extend(["--prompt-text", ref_text])

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(FISH_DIR)
        )
        if result.returncode != 0:
            raise RuntimeError(f"Text2Semantic error: {result.stderr[:500]}")

        # Find generated codes file
        codes_file = os.path.join(FISH_DIR, "codes_0.npy")
        if not os.path.exists(codes_file):
            raise RuntimeError("No se generaron tokens semánticos.")

        # Step 3: Decode tokens to audio
        logger.info("Step 3: Decoding to audio...")
        result = subprocess.run(
            [sys.executable,
             str(FISH_DIR / "fish_speech" / "models" / "dac" / "inference.py"),
             "-i", codes_file,
             "--checkpoint-path", str(checkpoint_dir / "codec.pth"),
             ],
            capture_output=True, text=True, cwd=str(FISH_DIR)
        )
        if result.returncode != 0:
            raise RuntimeError(f"Decoder error: {result.stderr[:500]}")

        # Read the output
        output_file = os.path.join(FISH_DIR, "fake.wav")
        if not os.path.exists(output_file):
            raise RuntimeError("No se generó archivo de audio.")

        import soundfile as sf
        wav, sr = sf.read(output_file)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        logger.info(f"Fish Audio: generated {len(wav)/sr:.1f}s at {sr}Hz")
        return wav, sr

    finally:
        # Cleanup temp files
        for f in ["fake.npy", "fake.wav", "codes_0.npy"]:
            p = os.path.join(FISH_DIR, f)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
