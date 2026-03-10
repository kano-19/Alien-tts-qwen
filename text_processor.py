"""
Text Processor - Smart chunking for long texts and emotion/breathing tag handling.
Splits long scripts into manageable chunks while preserving context and tags.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum characters per chunk (Qwen3-TTS can handle up to ~10 min audio, 
# but smaller chunks = less memory + faster processing)
DEFAULT_MAX_CHARS = 250

# Emotion/style preset mappings
EMOTION_PRESETS = {
    "[happy]": "Speak in a happy, cheerful tone",
    "[sad]": "Speak in a sad, melancholic tone",
    "[angry]": "Speak in an angry, forceful tone",
    "[whisper]": "Whisper softly",
    "[shout]": "Shout loudly with intensity",
    "[calm]": "Speak in a calm, soothing tone",
    "[excited]": "Speak with excitement and energy",
    "[scared]": "Speak in a frightened, trembling voice",
    "[laugh]": "Speak while laughing",
    "[cry]": "Speak while crying, with a breaking voice",
    "[breath]": "Speak with audible breathing, natural pauses",
    "[sigh]": "Speak with a deep sigh, tired tone",
    "[serious]": "Speak in a serious, formal tone",
    "[romantic]": "Speak in a warm, romantic, intimate tone",
    "[narrator]": "Speak as a professional narrator, clear and articulate",
    "[child]": "Speak like a young child, high-pitched",
    "[elderly]": "Speak like an elderly person, slow and wise",
    "[robotic]": "Speak in a flat, monotone, robotic manner",
}

# Pattern for break/pause markers
BREAK_PATTERN = re.compile(r'<break\s+time=["\'](\d+)(ms|s)["\']\s*/?>',  re.IGNORECASE)

# Pattern for emotion tags
EMOTION_TAG_PATTERN = re.compile(r'\[(\w+)\]')


@dataclass
class TextChunk:
    """A single processed text chunk ready for TTS."""
    text: str
    instruct: str = ""
    pause_after_ms: int = 0
    index: int = 0


@dataclass
class ProcessedScript:
    """Complete processed script with all chunks."""
    chunks: list[TextChunk] = field(default_factory=list)
    original_text: str = ""
    total_chars: int = 0
    num_chunks: int = 0


class TextProcessor:
    """Processes text for TTS: chunking, emotion tags, pauses."""

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS):
        self.max_chars = max_chars

    def process(self, text: str, default_instruct: str = "") -> ProcessedScript:
        """
        Process a full script into TTS-ready chunks.

        Args:
            text: The full text/script to process
            default_instruct: Default emotion/style instruction for all chunks

        Returns:
            ProcessedScript with list of TextChunks
        """
        original_text = text.strip()
        if not original_text:
            return ProcessedScript(
                chunks=[], original_text="", total_chars=0, num_chunks=0
            )

        # Extract global emotion tags at the start (if any)
        global_instruct = default_instruct

        # Split into paragraphs first
        paragraphs = self._split_paragraphs(original_text)

        # Process each paragraph into chunks
        all_chunks = []
        chunk_index = 0

        for para in paragraphs:
            # Check for emotion tags in this paragraph
            para_instruct, clean_para = self._extract_emotion_tags(para)
            if not para_instruct:
                para_instruct = global_instruct

            # Check for break markers
            clean_para, breaks = self._extract_breaks(clean_para)

            if not clean_para.strip():
                continue

            # Split paragraph into sentence-level chunks if too long
            sentences = self._split_sentences(clean_para)
            current_chunk_text = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # If adding this sentence exceeds max, flush current chunk
                if current_chunk_text and len(current_chunk_text) + len(sentence) + 1 > self.max_chars:
                    all_chunks.append(TextChunk(
                        text=current_chunk_text.strip(),
                        instruct=para_instruct,
                        pause_after_ms=0,
                        index=chunk_index,
                    ))
                    chunk_index += 1
                    current_chunk_text = ""

                # If single sentence is too long, force-split at word boundaries
                if len(sentence) > self.max_chars:
                    sub_chunks = self._force_split(sentence, self.max_chars)
                    for i, sub in enumerate(sub_chunks):
                        all_chunks.append(TextChunk(
                            text=sub.strip(),
                            instruct=para_instruct,
                            pause_after_ms=0,
                            index=chunk_index,
                        ))
                        chunk_index += 1
                else:
                    if current_chunk_text:
                        current_chunk_text += " " + sentence
                    else:
                        current_chunk_text = sentence

            # Flush remaining text in current chunk
            if current_chunk_text.strip():
                pause = breaks.get(len(all_chunks), 0)
                all_chunks.append(TextChunk(
                    text=current_chunk_text.strip(),
                    instruct=para_instruct,
                    pause_after_ms=pause,
                    index=chunk_index,
                ))
                chunk_index += 1

            # Add paragraph break pause
            if all_chunks:
                all_chunks[-1].pause_after_ms = max(
                    all_chunks[-1].pause_after_ms, 300
                )

        return ProcessedScript(
            chunks=all_chunks,
            original_text=original_text,
            total_chars=len(original_text),
            num_chunks=len(all_chunks),
        )

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs (double newline or more)."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using common delimiters."""
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(
            r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])$',
            text
        )
        return [s.strip() for s in sentences if s.strip()]

    def _force_split(self, text: str, max_len: int) -> list[str]:
        """Force-split a long sentence at word/clause boundaries."""
        chunks = []
        words = text.split()
        current = ""

        for word in words:
            if current and len(current) + len(word) + 1 > max_len:
                chunks.append(current)
                current = word
            else:
                current = current + " " + word if current else word

        if current:
            chunks.append(current)
        return chunks

    def _extract_emotion_tags(self, text: str) -> tuple[str, str]:
        """
        Extract emotion tags from text and return (instruct, clean_text).
        Tags like [happy], [sad], [breath] are converted to instruct strings.
        """
        found_tags = []
        clean = text

        for match in EMOTION_TAG_PATTERN.finditer(text):
            tag = f"[{match.group(1).lower()}]"
            if tag in EMOTION_PRESETS:
                found_tags.append(EMOTION_PRESETS[tag])
                clean = clean.replace(match.group(0), "", 1)

        instruct = ". ".join(found_tags) if found_tags else ""
        return instruct, clean.strip()

    def _extract_breaks(self, text: str) -> tuple[str, dict]:
        """
        Extract <break time="500ms"/> markers and return (clean_text, breaks_dict).
        """
        breaks = {}
        clean = text

        for match in BREAK_PATTERN.finditer(text):
            value = int(match.group(1))
            unit = match.group(2)
            if unit == "s":
                value *= 1000
            breaks[0] = value  # simplified: attach to chunk
            clean = clean.replace(match.group(0), " ", 1)

        return clean.strip(), breaks

    @staticmethod
    def get_emotion_presets() -> dict:
        """Get available emotion preset tags and their descriptions."""
        return EMOTION_PRESETS.copy()

    @staticmethod
    def get_emotion_help_text() -> str:
        """Get user-friendly help text for emotion tags."""
        lines = ["**Available Emotion Tags** (add to your text):"]
        for tag, desc in EMOTION_PRESETS.items():
            lines.append(f"  `{tag}` → {desc}")
        lines.append("")
        lines.append("**Pause marker:** `<break time=\"500ms\"/>` → insert silence")
        lines.append("")
        lines.append("**Or use free-text instruct** for custom styles.")
        return "\n".join(lines)
