"""
Alien TTS Qwen - Main Gradio Application
A comprehensive TTS studio with custom voices, voice design, voice cloning,
and audio upscaling powered by Qwen3-TTS.
Features: Blue premium UI, animated processing, modular engine selection.
"""

import os
import sys
import logging
import traceback

import numpy as np
import gradio as gr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AlienTTSQwen")
logging.getLogger("sox").setLevel(logging.ERROR)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_check import get_system_info, print_system_report, get_recommended_model_size
from model_manager import ModelManager
from text_processor import TextProcessor
import audio_upscaler
import audio_utils

# ─── Global State ───────────────────────────────────────────────────────────

system_info = get_system_info()
print_system_report(system_info)
recommended_size = get_recommended_model_size(system_info)
text_proc = TextProcessor()

_speakers_cache = None
_languages_cache = None


def get_speakers_list(model_size="1.7B"):
    global _speakers_cache
    if _speakers_cache is None:
        try:
            from engines import custom_voice
            _speakers_cache = custom_voice.get_speakers(model_size)
        except Exception:
            _speakers_cache = [
                "Vivian", "Ryan", "Chelsie", "Ethan", "Aria", "Benjamin",
                "Sophia", "Lucas", "Isabella", "Mason",
            ]
    return _speakers_cache


def get_languages_list(model_size="1.7B"):
    global _languages_cache
    if _languages_cache is None:
        try:
            from engines import custom_voice
            _languages_cache = custom_voice.get_languages(model_size)
        except Exception:
            _languages_cache = [
                "Auto", "Chinese", "English", "Japanese", "Korean",
                "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
            ]
    return _languages_cache


def get_vram_status():
    mgr = ModelManager()
    usage = mgr.get_vram_usage()
    loaded = mgr.get_loaded_models()
    mode = mgr.operating_mode
    mode_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴", "cpu": "⚪"}
    s = f"{mode_emoji.get(mode, '⚪')} **{mode.upper()}**"
    if usage["total_gb"] > 0:
        pct = (usage["allocated_gb"] / usage["total_gb"]) * 100
        s += f" — VRAM {usage['allocated_gb']:.1f}/{usage['total_gb']:.1f} GB ({pct:.0f}%)"
    if loaded:
        s += f" — Loaded: {', '.join(loaded)}"
    return s


# ─── Mode Descriptions ─────────────────────────────────────────────────────

MODE_INFO = {
    "🎤 Custom Voice": {
        "text_placeholder": "Escribe tu texto aquí...\n\nPuedes usar tags de emoción como:\n[happy] para alegre  [sad] para triste  [whisper] para susurrar\n[breath] para respiración  [angry] para enojado  [laugh] para risa\n\nEjemplo: [happy] ¡Hola! Estoy muy contento de conocerte.",
        "instruct_placeholder": "Instrucción de estilo/emoción (opcional):\nEj: 'Habla con mucha energía y entusiasmo'\nEj: 'Voz suave y relajada, como contando un cuento'",
        "description": "Genera voz usando los hablantes integrados de Qwen3-TTS. Selecciona un hablante, idioma y opcionalmente un estilo emocional.",
        "show_speaker": True,
        "show_ref_audio": False,
        "show_voice_desc": False,
        "show_ref_sentence": False,
    },
    "🎨 Voice Design": {
        "text_placeholder": "Escribe el texto que quieres que diga la voz diseñada...\n\nEjemplo: 'Buenos días, bienvenidos a nuestro programa de hoy.'",
        "instruct_placeholder": "Describe la voz que quieres crear:\nEj: 'Mujer, 28 años, voz cálida y suave, con un toque de misterio'\nEj: 'Hombre mayor, 65 años, voz grave y sabia, habla despacio'\nEj: 'Niño de 8 años, voz aguda y llena de curiosidad'",
        "description": "Crea una voz única desde cero usando una descripción en texto. Describe edad, género, tono, personalidad, acento...",
        "show_speaker": False,
        "show_ref_audio": False,
        "show_voice_desc": True,
        "show_ref_sentence": False,
    },
    "🎙️ Voice Clone": {
        "text_placeholder": "Escribe el texto que quieres que diga la voz clonada...\n\nEjemplo: 'Este es un texto de prueba con la voz clonada.'",
        "instruct_placeholder": "Transcripción del audio de referencia (recomendado):\nEscribe exactamente lo que dice el audio de referencia para mejor calidad.",
        "description": "Clona una voz a partir de un audio de referencia (3+ segundos). Sube un audio claro de la voz que quieres copiar.",
        "show_speaker": False,
        "show_ref_audio": True,
        "show_voice_desc": False,
        "show_ref_sentence": False,
    },
    "🔧 Design → Clone": {
        "text_placeholder": "Escribe todo el texto que quieres generar con la voz diseñada...\n\nTodo el texto se generará con la misma voz consistente.",
        "instruct_placeholder": "Describe la voz que quieres crear:\nEj: 'Mujer joven, 22 años, voz enérgica y clara, acento neutro'",
        "description": "Primero diseña una voz con una descripción, luego se clona automáticamente para mantener consistencia en textos largos.",
        "show_speaker": False,
        "show_ref_audio": False,
        "show_voice_desc": True,
        "show_ref_sentence": True,
        "show_fish_fields": False,
    },
    "🐟 Fish Audio S2 Pro": {
        "text_placeholder": "Escribe el texto que quieres generar con Fish Audio...\n\nSoporta tags como [laugh], [whispers], [super happy]\n\nEjemplo: [laugh] ¡Hola! Esto es increíble.",
        "instruct_placeholder": "Transcripción del audio de referencia (recomendado):\nEscribe lo que dice tu audio de referencia.",
        "description": "⚠️ LICENCIA CC-NC: Solo uso personal, NO comercial. TTS local con clonación de voz. Modelo S1-mini (~4GB VRAM). Se descarga automáticamente.",
        "show_speaker": False,
        "show_ref_audio": True,
        "show_voice_desc": False,
        "show_ref_sentence": False,
        "show_fish_fields": True,
    },
}

# Add show_fish_fields default to all modes that don't have it
for _mode_key in MODE_INFO:
    MODE_INFO[_mode_key].setdefault("show_fish_fields", False)

MODES = list(MODE_INFO.keys())


# ─── Generation dispatcher ─────────────────────────────────────────────────

def on_mode_change(mode):
    """Update UI visibility and placeholders when mode changes."""
    info = MODE_INFO[mode]
    return (
        gr.update(placeholder=info["text_placeholder"]),             # text
        gr.update(placeholder=info["instruct_placeholder"],
                  visible=not info["show_voice_desc"],
                  label="Instrucción de emoción / estilo" if not info["show_voice_desc"] else ""),
        gr.update(visible=info["show_speaker"]),                      # speaker
        gr.update(visible=info["show_ref_audio"]),                    # ref_audio
        gr.update(visible=info["show_voice_desc"],
                  placeholder=info["instruct_placeholder"] if info["show_voice_desc"] else ""),  # voice_desc
        gr.update(visible=info["show_ref_sentence"]),                 # ref_sentence
        f"**{mode}** — {info['description']}",                        # mode description
    )


def generate_tts(mode, text, instruct, speaker, language, model_size, upscale,
                  ref_audio, ref_text_transcript, voice_description, ref_sentence,
                  x_vector_only, progress=gr.Progress()):
    """Unified generation function that dispatches to the correct engine."""
    if not text.strip():
        return None, "❌ Ingresa texto para generar."

    try:
        if mode == "🎤 Custom Voice":
            return _gen_custom_voice(text, language, speaker, instruct, model_size, upscale, progress)
        elif mode == "🎨 Voice Design":
            return _gen_voice_design(text, language, voice_description, upscale, progress)
        elif mode == "🎙️ Voice Clone":
            return _gen_voice_clone(text, language, ref_audio, ref_text_transcript, x_vector_only, model_size, upscale, progress)
        elif mode == "🔧 Design → Clone":
            return _gen_design_clone(ref_sentence, voice_description, language, text, model_size, upscale, progress)
        elif mode == "🐟 Fish Audio S2 Pro":
            return _gen_fish_audio(text, ref_audio, ref_text_transcript, upscale, progress)
        else:
            return None, f"❌ Modo desconocido: {mode}"
    except Exception as e:
        logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {str(e)}"


def _process_chunks(engine_func, text, extra_kwargs, upscale, progress, start_pct=0.1, end_pct=0.9):
    """Common chunk processing pipeline for all engines."""
    script = text_proc.process(text)
    all_wavs = []
    pauses = []
    sr = 24000

    for i, chunk in enumerate(script.chunks):
        pct = start_pct + (end_pct - start_pct) * ((i + 1) / script.num_chunks)
        progress(pct, desc=f"🛸 Generando fragmento {i+1}/{script.num_chunks}...")
        wav, sr = engine_func(chunk, **extra_kwargs)
        all_wavs.append(wav)
        pauses.append(chunk.pause_after_ms)

    progress(0.92, desc="🔗 Uniendo fragmentos...")
    merged = audio_utils.merge_chunks(all_wavs, sr, pauses)

    if upscale:
        progress(0.95, desc="📈 Upscaling a 48kHz...")
        merged, sr = audio_upscaler.upscale(merged, sr, 48000)

    merged = audio_utils.normalize(merged)
    path = audio_utils.export(merged, sr)
    duration = audio_utils.get_duration_str(merged, sr)
    status = f"✅ {duration} | {sr}Hz | {script.num_chunks} fragmento(s) | Guardado: {os.path.basename(path)}"
    return (sr, merged), status


def _gen_custom_voice(text, language, speaker, instruct, model_size, upscale, progress):
    from engines import custom_voice
    progress(0.05, desc="🚀 Cargando modelo Custom Voice...")

    def engine_fn(chunk, **kw):
        return custom_voice.generate(
            text=chunk.text,
            language=kw["language"],
            speaker=kw["speaker"],
            instruct=chunk.instruct if chunk.instruct else kw.get("instruct", ""),
            model_size=kw["model_size"],
        )

    return _process_chunks(engine_fn, text,
                           {"language": language, "speaker": speaker, "instruct": instruct, "model_size": model_size},
                           upscale, progress)


def _gen_voice_design(text, language, voice_description, upscale, progress):
    from engines import voice_design
    progress(0.05, desc="🚀 Cargando modelo Voice Design...")

    def engine_fn(chunk, **kw):
        return voice_design.generate(
            text=chunk.text, language=kw["language"], instruct=kw["voice_description"],
        )

    return _process_chunks(engine_fn, text,
                           {"language": language, "voice_description": voice_description},
                           upscale, progress)


def _gen_voice_clone(text, language, ref_audio, ref_text, x_vector_only, model_size, upscale, progress):
    from engines import voice_clone

    if ref_audio is None:
        return None, "❌ Sube un audio de referencia."

    progress(0.05, desc="🚀 Cargando modelo Voice Clone...")
    progress(0.1, desc="🔍 Creando perfil de voz...")
    prompt = voice_clone.create_reusable_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text if ref_text.strip() and not x_vector_only else "",
        x_vector_only_mode=x_vector_only,
        model_size=model_size,
    )

    def engine_fn(chunk, **kw):
        return voice_clone.generate_with_prompt(
            text=chunk.text, language=kw["language"],
            voice_clone_prompt=kw["prompt"], model_size=kw["model_size"],
        )

    return _process_chunks(engine_fn, text,
                           {"language": language, "prompt": prompt, "model_size": model_size},
                           upscale, progress, start_pct=0.2)


def _gen_design_clone(ref_sentence, voice_description, language, text, model_size, upscale, progress):
    from engines import voice_design, voice_clone

    if not voice_description.strip():
        return None, "❌ Describe la voz que quieres crear."
    if not ref_sentence.strip():
        return None, "❌ Escribe una frase de referencia corta."

    progress(0.05, desc="🎨 Diseñando la voz...")
    ref_wav, sr = voice_design.generate(text=ref_sentence, language=language, instruct=voice_description)

    progress(0.15, desc="🔍 Clonando la voz diseñada...")
    prompt = voice_clone.create_reusable_prompt(ref_audio=(ref_wav, sr), ref_text=ref_sentence, model_size=model_size)

    def engine_fn(chunk, **kw):
        return voice_clone.generate_with_prompt(
            text=chunk.text, language=kw["language"],
            voice_clone_prompt=kw["prompt"], model_size=kw["model_size"],
        )

    return _process_chunks(engine_fn, text,
                           {"language": language, "prompt": prompt, "model_size": model_size},
                           upscale, progress, start_pct=0.2)


def _gen_fish_audio(text, ref_audio, ref_text, upscale, progress):
    from engines import fish_audio

    # Auto-install and download model on first use
    progress(0.05, desc="🐟 Verificando Fish Audio...")
    setup_msg = fish_audio.setup_if_needed(model_name="s1-mini", progress_callback=progress)
    if "❌" in setup_msg:
        return None, setup_msg

    progress(0.3, desc="🐟 Generando audio localmente con Fish Audio...")

    wav, sr = fish_audio.generate(
        text=text,
        ref_audio_path=ref_audio if ref_audio else None,
        ref_text=ref_text if ref_text else "",
        model_name="s1-mini",
    )

    if upscale and sr < 48000:
        progress(0.9, desc="📈 Upscaling a 48kHz...")
        wav, sr = audio_upscaler.upscale(wav, sr, 48000)

    wav = audio_utils.normalize(wav)
    path = audio_utils.export(wav, sr)
    duration = audio_utils.get_duration_str(wav, sr)
    status = f"✅ {duration} | {sr}Hz | 🐟 Fish Audio S1-mini (local) | Guardado: {os.path.basename(path)}"
    return (sr, wav), status


def install_fish_audio_manual():
    """Manual install trigger from the UI."""
    from engines import fish_audio
    msg1 = fish_audio.install_fish_speech()
    if "❌" in msg1:
        return msg1
    msg2 = fish_audio.download_model("s1-mini")
    return f"{msg1} | {msg2}"


def unload_all_models():
    mgr = ModelManager()
    mgr.unload_all()
    return get_vram_status(), "✅ Todos los modelos descargados de la memoria."


# ─── CSS Theme ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', system-ui, sans-serif !important; }

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

/* ── Header ── */
.header-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0d2847 30%, #0f3460 60%, #1a5276 100%);
    border-radius: 20px;
    padding: 32px 36px 28px;
    margin-bottom: 16px;
    border: 1px solid rgba(59, 130, 246, 0.25);
    box-shadow: 0 0 60px rgba(37, 99, 235, 0.15), 0 8px 32px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.header-banner h1 {
    color: #fff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 4px 0 !important;
    background: linear-gradient(90deg, #93c5fd, #60a5fa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-banner p {
    color: rgba(147, 197, 253, 0.8) !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
    font-weight: 400;
}

/* ── Status bar ── */
.status-bar {
    background: linear-gradient(135deg, rgba(15,25,50,0.8), rgba(20,35,65,0.7));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 0.85rem;
}

/* ── Mode description ── */
.mode-info {
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: rgba(147, 197, 253, 0.9);
}

/* ── Tips box ── */
.tips-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(59, 130, 246, 0.15);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.82rem;
    line-height: 1.7;
}

/* ── Processing animation ── */
@keyframes ufo-float {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    25% { transform: translateY(-8px) rotate(3deg); }
    50% { transform: translateY(-2px) rotate(-2deg); }
    75% { transform: translateY(-10px) rotate(1deg); }
}
@keyframes beam-pulse {
    0%, 100% { opacity: 0.3; transform: scaleY(1); }
    50% { opacity: 0.8; transform: scaleY(1.1); }
}
@keyframes stars-twinkle {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}
@keyframes progress-glow {
    0% { box-shadow: 0 0 5px rgba(59,130,246,0.3); }
    50% { box-shadow: 0 0 20px rgba(59,130,246,0.6), 0 0 40px rgba(59,130,246,0.2); }
    100% { box-shadow: 0 0 5px rgba(59,130,246,0.3); }
}

.processing-animation {
    text-align: center;
    padding: 20px;
    min-height: 100px;
}
.ufo-scene {
    display: inline-block;
    position: relative;
    width: 200px;
    height: 100px;
}
.ufo {
    font-size: 48px;
    animation: ufo-float 3s ease-in-out infinite;
    display: inline-block;
    filter: drop-shadow(0 0 12px rgba(59,130,246,0.5));
}
.beam {
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 35px;
    background: linear-gradient(180deg, rgba(59,130,246,0.4) 0%, transparent 100%);
    clip-path: polygon(20% 0%, 80% 0%, 100% 100%, 0% 100%);
    animation: beam-pulse 1.5s ease-in-out infinite;
}
.stars {
    font-size: 10px;
    position: absolute;
    color: rgba(147,197,253,0.6);
    animation: stars-twinkle 2s ease-in-out infinite;
}

/* ── Progress styling ── */
.progress-wrap .wrap {
    animation: progress-glow 2s ease-in-out infinite;
}

/* ── License warning ── */
.license-warning {
    background: rgba(180, 83, 9, 0.15) !important;
    border: 1px solid rgba(251, 191, 36, 0.4) !important;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #fbbf24 !important;
}

footer { display: none !important; }
"""

# ─── Animated Processing HTML ──────────────────────────────────────────────

PROCESSING_IDLE_HTML = """
<div style="text-align:center; padding:16px; opacity:0.5;">
    <span style="font-size:36px;">🛸</span>
    <p style="color:rgba(147,197,253,0.7); font-size:0.85rem; margin:8px 0 0;">
        Listo para generar — presiona el botón
    </p>
</div>
"""

PROCESSING_ACTIVE_HTML = """
<div class="processing-animation">
    <div class="ufo-scene">
        <span class="stars" style="top:5px;left:15px;">✦</span>
        <span class="stars" style="top:15px;right:20px;animation-delay:0.7s;">✧</span>
        <span class="stars" style="top:8px;left:60%;animation-delay:1.3s;">✦</span>
        <div class="ufo">🛸</div>
        <div class="beam"></div>
    </div>
    <p style="color:#60a5fa; font-size:0.9rem; margin:8px 0 4px; font-weight:600;">
        Procesando con IA...
    </p>
    <p style="color:rgba(147,197,253,0.7); font-size:0.8rem; margin:0;">
        La nave está generando tu audio 🔊
    </p>
</div>
"""

PROCESSING_DONE_HTML = """
<div style="text-align:center; padding:16px;">
    <span style="font-size:36px;">✅</span>
    <p style="color:#34d399; font-size:0.9rem; margin:8px 0 0; font-weight:600;">
        ¡Audio generado exitosamente!
    </p>
</div>
"""


# ─── Build Gradio UI ───────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Alien TTS Qwen",
        theme=gr.themes.Base(
            primary_hue=gr.themes.Color(
                c50="#eff6ff", c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd",
                c400="#60a5fa", c500="#3b82f6", c600="#2563eb", c700="#1d4ed8",
                c800="#1e40af", c900="#1e3a8a", c950="#172554",
            ),
            secondary_hue="blue",
            neutral_hue="slate",
            font=("Inter", "system-ui", "sans-serif"),
        ).set(
            body_background_fill="#0b1120",
            body_background_fill_dark="#0b1120",
            block_background_fill="rgba(15, 23, 42, 0.85)",
            block_background_fill_dark="rgba(15, 23, 42, 0.85)",
            block_border_width="1px",
            block_border_color="rgba(59, 130, 246, 0.2)",
            block_border_color_dark="rgba(59, 130, 246, 0.2)",
            block_label_text_color="rgba(147, 197, 253, 0.9)",
            block_label_text_color_dark="rgba(147, 197, 253, 0.9)",
            block_title_text_color="#93c5fd",
            block_title_text_color_dark="#93c5fd",
            body_text_color="rgba(203, 213, 225, 0.95)",
            body_text_color_dark="rgba(203, 213, 225, 0.95)",
            input_background_fill="rgba(15, 23, 42, 0.95)",
            input_background_fill_dark="rgba(15, 23, 42, 0.95)",
            input_border_color="rgba(59, 130, 246, 0.3)",
            input_border_color_dark="rgba(59, 130, 246, 0.3)",
            input_placeholder_color="rgba(100, 130, 180, 0.5)",
            input_placeholder_color_dark="rgba(100, 130, 180, 0.5)",
            button_primary_background_fill="linear-gradient(135deg, #1d4ed8, #2563eb, #3b82f6)",
            button_primary_background_fill_dark="linear-gradient(135deg, #1d4ed8, #2563eb, #3b82f6)",
            button_primary_background_fill_hover="linear-gradient(135deg, #2563eb, #3b82f6, #60a5fa)",
            button_primary_background_fill_hover_dark="linear-gradient(135deg, #2563eb, #3b82f6, #60a5fa)",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="rgba(30, 58, 138, 0.4)",
            button_secondary_background_fill_dark="rgba(30, 58, 138, 0.4)",
            button_secondary_text_color="#93c5fd",
            button_secondary_text_color_dark="#93c5fd",
            checkbox_background_color="rgba(15, 23, 42, 0.9)",
            checkbox_background_color_dark="rgba(15, 23, 42, 0.9)",
            shadow_drop="0 4px 16px rgba(0,0,0,0.3)",
            shadow_drop_lg="0 8px 32px rgba(0,0,0,0.4)",
            block_radius="14px",
            button_large_radius="12px",
        ),
        css=CUSTOM_CSS,
    ) as demo:

        # ── Header ──
        gr.HTML("""
            <div class="header-banner">
                <h1>🛸 Alien TTS Qwen</h1>
                <p>Generación de voz con IA · Voces personalizadas · Clonación · Diseño de voz · Upscaling</p>
            </div>
        """)

        # ── Status Bar ──
        with gr.Row():
            vram_display = gr.Markdown(get_vram_status(), elem_classes=["status-bar"])

        with gr.Row():
            with gr.Column(scale=3):
                # ── Mode Selector ──
                mode_selector = gr.Dropdown(
                    choices=MODES,
                    value=MODES[0],
                    label="🎯 Modo de Generación",
                    info="Selecciona cómo quieres generar el audio",
                    interactive=True,
                )
                mode_desc = gr.Markdown(
                    f"**{MODES[0]}** — {MODE_INFO[MODES[0]]['description']}",
                    elem_classes=["mode-info"],
                )

            with gr.Column(scale=1):
                with gr.Row():
                    language = gr.Dropdown(
                        choices=get_languages_list(),
                        value="Auto",
                        label="🌍 Idioma",
                        scale=1,
                    )
                    model_size = gr.Dropdown(
                        choices=["1.7B", "0.6B"],
                        value=recommended_size,
                        label="📦 Modelo",
                        info=f"Recomendado: {recommended_size}",
                        scale=1,
                    )

        # ── Main Content ──
        with gr.Row():
            # LEFT: Inputs
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="📝 Texto a Generar",
                    placeholder=MODE_INFO[MODES[0]]["text_placeholder"],
                    lines=8,
                    max_lines=30,
                )

                # Conditional fields
                speaker = gr.Dropdown(
                    choices=get_speakers_list(),
                    value=get_speakers_list()[0] if get_speakers_list() else "Vivian",
                    label="🗣️ Hablante",
                    visible=True,
                )
                instruct_input = gr.Textbox(
                    label="🎭 Instrucción de Emoción / Estilo",
                    placeholder=MODE_INFO[MODES[0]]["instruct_placeholder"],
                    lines=2,
                    visible=True,
                )
                voice_desc = gr.Textbox(
                    label="🎨 Descripción de la Voz",
                    placeholder="Describe la voz que quieres crear con detalle...",
                    lines=3,
                    visible=False,
                )
                ref_sentence = gr.Textbox(
                    label="📌 Frase de Referencia (para diseño)",
                    placeholder="Escribe una frase corta que se usará para crear la muestra de la voz diseñada...",
                    lines=2,
                    visible=False,
                )
                ref_audio = gr.Audio(
                    label="🎧 Audio de Referencia (3+ segundos)",
                    type="filepath",
                    visible=False,
                )
                ref_text_transcript = gr.Textbox(
                    label="📋 Transcripción del Audio de Referencia",
                    placeholder="Escribe exactamente lo que dice el audio de referencia...",
                    lines=2,
                    visible=False,
                )
                x_vector_only = gr.Checkbox(
                    label="Solo X-Vector (no necesita transcripción, menor calidad)",
                    value=False,
                    visible=False,
                )

                # Fish Audio specific fields
                fish_license_warning = gr.Markdown(
                    "⚠️ **LICENCIA NO COMERCIAL:** Fish Audio está bajo licencia "
                    "Research / CC-NC. Solo para uso personal. El uso comercial requiere "
                    "licencia directa de Fish Audio.\n\n"
                    "💻 **Modo local:** El modelo se instala y descarga automáticamente "
                    "la primera vez que generas audio. Tamaño: ~2GB.",
                    elem_classes=["license-warning"],
                    visible=False,
                )
                with gr.Row(visible=False) as fish_api_row:
                    fish_install_btn = gr.Button(
                        "🐟 Instalar Fish Audio (local)",
                        variant="secondary", size="sm", scale=1,
                    )

                with gr.Row():
                    upscale = gr.Checkbox(
                        label="📈 Upscale a 48kHz (mejor calidad)",
                        value=False,
                    )
                    unload_btn = gr.Button("🗑️ Liberar Memoria", variant="secondary", size="sm")

                generate_btn = gr.Button(
                    "🛸 Generar Audio",
                    variant="primary",
                    size="lg",
                )

            # RIGHT: Output + Animation
            with gr.Column(scale=2):
                animation_display = gr.HTML(PROCESSING_IDLE_HTML)
                audio_output = gr.Audio(label="🔊 Audio Generado", type="numpy")
                status_output = gr.Markdown("", elem_classes=["status-bar"])

                gr.Markdown(
                    "**📖 Guía Rápida:**\n\n"
                    "**Tags de emoción** (agrégalos al texto):\n"
                    "`[happy]` `[sad]` `[angry]` `[whisper]`\n"
                    "`[breath]` `[sigh]` `[laugh]` `[cry]`\n"
                    "`[excited]` `[calm]` `[narrator]`\n\n"
                    "**Pausas:** `<break time=\"500ms\"/>`\n\n"
                    "**Tip:** Textos largos se dividen automáticamente en fragmentos para mejor rendimiento.",
                    elem_classes=["tips-box"],
                )

        # ── Mode change handler ──
        def on_mode_changed(mode):
            info = MODE_INFO[mode]
            is_fish = info.get("show_fish_fields", False)
            is_clone = info["show_ref_audio"] and not is_fish
            updates = [
                gr.update(placeholder=info["text_placeholder"]),
                gr.update(placeholder=info["instruct_placeholder"],
                         visible=not info["show_voice_desc"] and not is_fish),
                gr.update(visible=info["show_speaker"]),
                gr.update(visible=info["show_ref_audio"]),
                gr.update(visible=info["show_voice_desc"],
                         placeholder=info["instruct_placeholder"] if info["show_voice_desc"] else ""),
                gr.update(visible=info["show_ref_sentence"]),
                f"**{mode}** — {info['description']}",
                # Show ref transcript for clone + fish modes
                gr.update(visible=info["show_ref_audio"]),
                gr.update(visible=is_clone),
                # Fish Audio fields
                gr.update(visible=is_fish),
                gr.update(visible=is_fish),
                # Hide Idioma/Modelo for Fish Audio (not needed)
                gr.update(visible=not is_fish),
                gr.update(visible=not is_fish),
            ]
            return updates

        mode_selector.change(
            fn=on_mode_changed,
            inputs=[mode_selector],
            outputs=[text_input, instruct_input, speaker, ref_audio, voice_desc,
                     ref_sentence, mode_desc, ref_text_transcript, x_vector_only,
                     fish_license_warning, fish_api_row,
                     language, model_size],
        )

        # ── Fish install button ──
        fish_install_btn.click(
            fn=install_fish_audio_manual,
            outputs=[status_output],
        )

        # ── Generate handler ──
        def on_generate_start():
            return PROCESSING_ACTIVE_HTML

        def on_generate_end(audio, status):
            html = PROCESSING_DONE_HTML if audio is not None else PROCESSING_IDLE_HTML
            return html

        generate_btn.click(
            fn=on_generate_start,
            outputs=[animation_display],
        ).then(
            fn=generate_tts,
            inputs=[mode_selector, text_input, instruct_input, speaker, language, model_size,
                    upscale, ref_audio, ref_text_transcript, voice_desc, ref_sentence, x_vector_only],
            outputs=[audio_output, status_output],
        ).then(
            fn=on_generate_end,
            inputs=[audio_output, status_output],
            outputs=[animation_display],
        )

        # ── Unload handler ──
        unload_btn.click(
            fn=unload_all_models,
            outputs=[vram_display, status_output],
        )

    return demo


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        inbrowser=True,
    )
