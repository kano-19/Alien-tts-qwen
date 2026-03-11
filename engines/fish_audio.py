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

# ── Subprocess timeout (seconds) ──
SUBPROCESS_TIMEOUT = 120

# ── State ────────────────────────────────────────────────────────────────

_installed = None
_model_loaded = False
_model_variant = None  # "s1-mini" or "s2-pro"


def _get_python_exe() -> str:
    """Get the correct Python executable (venv first, fallback to sys.executable)."""
    venv_python = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


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
        [_get_python_exe(), "-m", "pip", "install", "-e", ".", "--quiet"],
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
        [_get_python_exe(), "-m", "huggingface_hub.cli.main", "download",
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
        progress_callback(0.8, desc=f"✅ Modelo {model_name} listo! Parcheando dependencias...")

    # Fix: The s1-mini model lacks a tokenizer config on HF, causing transformers to crash
    if "s1-mini" in model_name:
        _patch_s1_tokenizer(str(checkpoint_dir))

    return f"✅ Modelo {model_name} descargado en {checkpoint_dir} y parcheado"

def _patch_s1_tokenizer(checkpoint_dir: str):
    """
    Downloads missing Qwen2 tokenizer files and injects Fish Audio semantic tokens 
    so Transformers can instantiate the tokenizer without raising a 'dual_ar' architecture KeyError.
    """
    tk_json = os.path.join(checkpoint_dir, "tokenizer.json")
    if os.path.exists(tk_json) and os.path.getsize(tk_json) > 1000:
        return # Already patched

    logger.info("Patching missing tokenizer files for S1-mini...")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import json

    try:
        # Download working tokenizer config from Qwen
        snapshot_download(
            repo_id="Qwen/Qwen2-7B-Instruct",
            allow_patterns=["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"],
            local_dir=checkpoint_dir,
        )

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        
        # Inject Fish Audio's 4096 semantic tokens dynamically
        special_tk_path = os.path.join(checkpoint_dir, "special_tokens.json")
        if os.path.exists(special_tk_path):
            with open(special_tk_path, "r", encoding="utf-8") as f:
                special_tokens = json.load(f)
            
            sorted_items = sorted(special_tokens.items(), key=lambda item: item[1])
            tokens_to_add = [item[0] for item in sorted_items]
            tokenizer.add_tokens(tokens_to_add)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Successfully injected {len(tokens_to_add)} semantic tokens into tokenizer -> vocab size: {len(tokenizer)}")
    except Exception as e:
        logger.error(f"Failed to patch tokenizer: {e}")


def setup_if_needed(model_name: str = "s1-mini", progress_callback=None, hf_token: str = "") -> str:
    """Install fish-speech and download model if not already done."""
    status_parts = []

    if not is_installed():
        msg = install_fish_speech(progress_callback)
        status_parts.append(msg)
        if "❌" in msg:
            return msg

    checkpoint_dir = CHECKPOINTS_DIR / model_name
    if not checkpoint_dir.exists() or not (checkpoint_dir / "codec.pth").exists():
        msg = download_model(model_name, progress_callback, hf_token=hf_token)
        status_parts.append(msg)
        if "❌" in msg:
            return msg

    return " | ".join(status_parts) if status_parts else "✅ Todo listo."


def _estimate_max_tokens(text: str) -> int:
    """
    Estimate a reasonable max token budget for the given text.
    
    Fish Speech generates ~12 audio tokens per second of speech.
    Average speech rate is ~3 words/second in Spanish.
    So: tokens ≈ (word_count / 3) * 12 = word_count * 4
    
    We add a generous 2x safety margin and clamp to [50, 512].
    """
    word_count = len(text.split())
    estimated = word_count * 8  # ~8 tokens per word with margin
    return max(50, min(estimated, 512))


def generate(
    text: str,
    ref_audio_path: Optional[str] = None,
    ref_text: str = "",
    model_name: str = "s1-mini",
) -> tuple[np.ndarray, int]:
    """
    Generate speech using local Fish Speech inference.

    Pipeline (single step with --output flag):
      1. Encode reference audio → VQ tokens (if cloning, via --prompt-audio)
      2. Generate semantic tokens from text AND decode to WAV in one pass

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

    python_exe = _get_python_exe()

    # Create a temp directory for all outputs
    output_dir = tempfile.mkdtemp(prefix="fish_")
    output_wav = os.path.join(output_dir, "generated.wav")

    try:
        # Calculate smart token budget based on text length
        max_tokens = _estimate_max_tokens(text)
        logger.info(f"Text length: {len(text)} chars, {len(text.split())} words → max_tokens={max_tokens}")

        # Build the inference command — single step: text → wav
        # Using --output flag makes inference.py decode directly to audio
        cmd = [
            python_exe,
            str(FISH_DIR / "fish_speech" / "models" / "text2semantic" / "inference.py"),
            "--text", text,
            "--max-new-tokens", str(max_tokens),
            "--half",  # Use FP16 for lower VRAM
            "--checkpoint-path", str(checkpoint_dir),
            "--output-dir", output_dir,
            "--output", output_wav,  # This triggers direct WAV output
        ]

        # Voice cloning: pass reference audio directly
        if ref_audio_path and os.path.exists(ref_audio_path):
            logger.info("Adding reference audio for voice cloning...")
            cmd.extend(["--prompt-audio", ref_audio_path])
            if ref_text.strip():
                cmd.extend(["--prompt-text", ref_text.strip()])

        logger.info(f"Running Fish Audio inference (timeout={SUBPROCESS_TIMEOUT}s)...")
        logger.info(f"Command: {' '.join(cmd[:6])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(FISH_DIR),
            timeout=SUBPROCESS_TIMEOUT,
        )

        # Log stderr for debugging (it contains loguru output)
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-5:]:
                logger.info(f"[fish-speech] {line.strip()}")

        if result.returncode != 0:
            # Extract the most useful error info
            stderr_tail = result.stderr[-500:] if result.stderr else "No error output"
            raise RuntimeError(f"Fish Audio inference failed (code {result.returncode}): {stderr_tail}")

        # Check if WAV was generated directly
        if os.path.exists(output_wav):
            import soundfile as sf
            wav, sr = sf.read(output_wav)
            wav = wav.astype(np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            logger.info(f"Fish Audio: generated {len(wav)/sr:.1f}s at {sr}Hz (direct decode)")
            return wav, sr

        # Fallback: check for codes_0.npy and decode separately
        codes_file = os.path.join(output_dir, "codes_0.npy")
        if not os.path.exists(codes_file):
            raise RuntimeError(
                "Fish Audio no generó audio ni tokens. "
                "Puede que el texto sea demasiado corto o el modelo no soporte el idioma."
            )

        logger.info("Direct decode not available, falling back to codec decoder...")
        
        # Step 2: Decode tokens to audio using codec
        fallback_wav = os.path.join(output_dir, "decoded.wav")
        codec_cmd = [
            python_exe,
            str(FISH_DIR / "fish_speech" / "models" / "dac" / "inference.py"),
            "--input-path", codes_file,
            "--output-path", fallback_wav,
            "--checkpoint-path", str(checkpoint_dir / "codec.pth"),
        ]

        result = subprocess.run(
            codec_cmd,
            capture_output=True,
            text=True,
            cwd=str(FISH_DIR),
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Codec decoder error: {result.stderr[-300:]}")

        if not os.path.exists(fallback_wav):
            raise RuntimeError("El decodificador de audio no generó archivo WAV.")

        import soundfile as sf
        wav, sr = sf.read(fallback_wav)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        logger.info(f"Fish Audio: generated {len(wav)/sr:.1f}s at {sr}Hz (codec fallback)")
        return wav, sr

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Fish Audio tardó más de {SUBPROCESS_TIMEOUT}s y fue cancelado. "
            "Intenta con un texto más corto."
        )

    finally:
        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass
