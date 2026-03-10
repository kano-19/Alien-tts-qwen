# 🛸 Alien TTS Qwen

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Qwen3--TTS-v1.7B%20%7C%200.6B-blue?logo=huggingface" />
  <img src="https://img.shields.io/badge/Gradio-5.0+-blue?logo=gradio" />
  <img src="https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  <b>Alien TTS Qwen</b> — Estudio de Text-to-Speech con IA de última generación. Genera, diseña y clona voces con una interfaz visual premium.
</p>

---

## ✨ Características

| Característica | Descripción |
|---|---|
| 🎤 **Custom Voice** | Voces integradas de alta calidad con control de emociones |
| 🎨 **Voice Design** | Crea voces únicas describiendo edad, género, tono, personalidad |
| 🎙️ **Voice Clone** | Clona cualquier voz con solo 3 segundos de audio de referencia |
| 🔧 **Design → Clone** | Diseña una voz y clónala para consistencia en textos largos |
| 📈 **Audio Upscaling** | Mejora de 24kHz a 48kHz con realce espectral |
| 🧠 **VRAM Inteligente** | Detecta tu GPU y adapta el uso de memoria automáticamente |
| ✂️ **Auto-Chunking** | Textos largos se dividen y generan por partes, luego se unen |
| 🎭 **Emociones** | Tags como `[happy]`, `[whisper]`, `[breath]` para control emocional |
| 🛸 **UI Premium** | Interfaz azul con animación de nave espacial durante el procesamiento |

---

## 🖥️ Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|---|---|---|
| **OS** | Windows 10/11 | Windows 10/11 |
| **Python** | 3.12 | 3.12 |
| **GPU** | NVIDIA 4GB VRAM | NVIDIA 8GB+ VRAM |
| **RAM** | 8 GB | 16 GB+ |
| **CUDA** | 12.1+ | 12.4 |
| **Disco** | 10 GB libres | 20 GB libres |

> **Nota:** Funciona sin GPU (modo CPU), pero la generación será lenta. Se recomienda GPU NVIDIA.

---

## 🚀 Instalación

### Opción 1: Automática (recomendada)

```bash
git clone https://github.com/kano-19/Alien-tts-qwen.git
cd Alien-tts-qwen
install.bat
```

El instalador automáticamente:
- ✅ Busca Python 3.12 en tu sistema
- ✅ Si no existe, intenta instalarlo vía `winget`
- ✅ Crea un entorno virtual aislado (`venv/`)
- ✅ Instala PyTorch con soporte CUDA
- ✅ Instala todas las dependencias
- ✅ Intenta instalar FlashAttention 2 (opcional, para más velocidad)
- ✅ Muestra un reporte de tu hardware

### Opción 2: Manual

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## ▶️ Uso

### Iniciar el servidor

```bash
start.bat
```

O manualmente:

```bash
venv\Scripts\activate
python app.py
```

Abre tu navegador en **http://localhost:7860**

### Flujo de uso

1. **Selecciona un modo** en el dropdown (Custom Voice, Voice Design, etc.)
2. **Escribe el texto** que quieres generar
3. **Configura** idioma, hablante, emoción, etc.
4. **Click en 🛸 Generar Audio**
5. **Escucha y descarga** el audio generado

---

## 🎯 Modos de Generación

### 🎤 Custom Voice
Usa las voces pre-entrenadas de Qwen3-TTS. Selecciona un hablante de la lista y opcionalmente agrega instrucciones de emoción.

```
Hablantes disponibles: Vivian, Ryan, Chelsie, Ethan, Aria, Benjamin, etc.
```

**Ejemplo de uso:**
- Texto: `[happy] ¡Hola! Bienvenidos a nuestro canal.`
- Hablante: `Vivian`
- Idioma: `Spanish`

### 🎨 Voice Design
Crea una voz completamente nueva describiendo sus características en texto natural.

**Ejemplo de descripción:**
> "Mujer, 28 años, voz cálida y profesional, tono medio, habla con confianza y un toque de calidez. Acento neutro latinoamericano."

### 🎙️ Voice Clone
Clona cualquier voz a partir de un audio de referencia de 3+ segundos.

**Requisitos:**
- Audio claro, sin ruido de fondo
- Un solo hablante
- Preferiblemente formato WAV
- Mejor calidad si incluyes la transcripción del audio

### 🔧 Design → Clone
Combina los dos anteriores: primero diseña una voz única, luego la clona para mantener consistencia en textos largos.

**Flujo:**
1. Describe la voz que quieres
2. Escribe una frase de muestra
3. El sistema genera una referencia con Voice Design
4. Luego clona esa voz para tu texto completo

---

## 🎭 Tags de Emoción

Inserta estos tags en tu texto para controlar las emociones:

| Tag | Efecto |
|-----|--------|
| `[happy]` | Tono alegre y animado |
| `[sad]` | Tono triste y melancólico |
| `[angry]` | Tono enojado y fuerte |
| `[whisper]` | Susurro suave |
| `[shout]` | Grito con intensidad |
| `[breath]` | Respiración audible, pausas naturales |
| `[sigh]` | Suspiro profundo |
| `[laugh]` | Habla mientras ríe |
| `[cry]` | Habla con voz quebrada |
| `[calm]` | Tono calmado y relajante |
| `[excited]` | Voz con energía y emoción |
| `[narrator]` | Narrador profesional, claro |
| `[romantic]` | Tono cálido e íntimo |

**Pausas:** Usa `<break time="500ms"/>` para insertar silencios.

---

## 🧠 Modelos Utilizados

### Qwen3-TTS (Alibaba Cloud / QwenLM)

| Modelo | Parámetros | VRAM | Uso |
|--------|-----------|------|-----|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | ~4 GB | Voces integradas con emociones |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | ~1.5 GB | Versión ligera de CustomVoice |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | ~4 GB | Diseño de voces desde texto |
| `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | ~4 GB | Clonación de voz (alta calidad) |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | ~1.5 GB | Clonación de voz (ligera) |
| `Qwen3-TTS-Tokenizer-12Hz` | — | Mínimo | Tokenizador de audio |

> Los modelos se descargan automáticamente desde Hugging Face la primera vez que se usan.

### Características técnicas de Qwen3-TTS
- **Arquitectura:** LM multi-codebook discreto end-to-end
- **Tokenizador:** Qwen3-TTS-Tokenizer-12Hz (compresión acústica eficiente)
- **Idiomas:** 10 idiomas (Chino, Inglés, Japonés, Coreano, Alemán, Francés, Ruso, Portugués, Español, Italiano)
- **Latencia:** Streaming con latencia tan baja como 97ms
- **Clonación:** Solo 3 segundos de audio de referencia necesarios
- **Licencia:** Apache 2.0

---

## 📈 Audio Upscaling

El upscaler convierte audio de **24kHz a 48kHz** usando:

1. **Resampling polifásico** de alta calidad (`scipy.signal.resample_poly`)
2. **Realce espectral** opcional: boost sutil de frecuencias altas (4-12kHz) para mayor claridad

> Es un proceso ligero sin descargas adicionales. Activable con un checkbox en la UI.

---

## 🔧 Gestión de Memoria (VRAM)

El sistema detecta automáticamente tu GPU y adapta su comportamiento:

| VRAM Disponible | Modo | Comportamiento |
|-----------------|------|----------------|
| ≥ 8 GB | 🟢 HIGH | Modelos permanecen cargados (más rápido) |
| 5-8 GB | 🟡 MEDIUM | 1 modelo en caché a la vez |
| < 5 GB | 🔴 LOW | Carga → usa → descarga cada vez |
| Sin GPU | ⚪ CPU | Funcional pero lento |

**Liberar memoria:** Usa el botón "🗑️ Liberar Memoria" en la UI para descargar todos los modelos de la VRAM.

---

## 📁 Estructura del Proyecto

```
alien-tts-qwen/
│
├── install.bat              # Instalador automático
├── start.bat                # Lanzador del servidor
├── requirements.txt         # Dependencias Python
│
├── app.py                   # Interfaz Gradio (UI principal)
├── system_check.py          # Detección de hardware (GPU/VRAM/RAM)
├── model_manager.py         # Gestión modular de modelos
├── text_processor.py        # Chunking inteligente + tags de emoción
├── audio_upscaler.py        # Upscaling 24kHz → 48kHz
├── audio_utils.py           # Merge, normalización, exportación
│
├── engines/                 # Motores TTS (modular)
│   ├── __init__.py
│   ├── custom_voice.py      # Motor Custom Voice
│   ├── voice_design.py      # Motor Voice Design
│   └── voice_clone.py       # Motor Voice Clone
│
└── outputs/                 # Audio generado (auto-creado)
```

### Diseño Modular

Cada archivo es independiente. Si algo falla en un motor, los demás siguen funcionando:

- **`system_check.py`** → Solo detecta hardware, no depende de nada
- **`model_manager.py`** → Singleton que maneja carga/descarga de modelos
- **`text_processor.py`** → Solo procesa texto, no necesita modelos
- **`audio_upscaler.py`** → Solo scipy, sin modelos pesados
- **`engines/*.py`** → Cada motor es un archivo separado

---

## 🔄 Auto-Chunking (Textos Largos)

Para textos largos, el sistema automáticamente:

1. **Divide** el texto en fragmentos de ~250 caracteres
2. **Respeta** límites de oraciones (no corta a mitad de frase)
3. **Preserva** tags de emoción para cada fragmento
4. **Genera** cada fragmento por separado (menos presión en VRAM)
5. **Une** todos los fragmentos con pausas naturales
6. **Aplica** normalización y opcionalmente upscaling

Esto permite generar textos de cualquier longitud sin quedarse sin memoria.

---

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b mi-mejora`
3. Haz tus cambios
4. Commit: `git commit -m "Agrega nueva funcionalidad"`
5. Push: `git push origin mi-mejora`
6. Abre un Pull Request

---

## 📄 Licencia

Este proyecto es de código abierto. Los modelos Qwen3-TTS están bajo licencia [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## 🙏 Créditos

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** — Alibaba Cloud / QwenLM Team
- **[Gradio](https://gradio.app)** — Interfaz web
- **[PyTorch](https://pytorch.org)** — Framework de deep learning
- **[Hugging Face](https://huggingface.co)** — Hosting de modelos
