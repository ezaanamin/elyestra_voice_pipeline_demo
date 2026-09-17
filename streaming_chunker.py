"""
Streaming Text Chunker for Elyestra Voice Pipeline
===================================================
Handles real-time token streams from an LLM and partitions text into optimal
audio synthesis chunks.

Key Responsibilities:
1. Punctuation-based sentence boundary detection using lookbehind regex.
2. Word-count caps (max_tts_words) to keep TTS inference chunks small and uniform.
3. Buffer-overflow protection (max_buffer_words) for run-on sentences or code blocks.
4. Stream finalization and tail flushing.
"""

import re
from typing import Callable, List, Optional


class StreamingChunker:
    """
    Buffers streamed tokens from an LLM and invokes a callback whenever a complete
    sentence or maximum word threshold is reached.
    """

    def __init__(
        self,
        max_tts_words: int = 15,
        max_buffer_words: int = 30,
        on_chunk: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Args:
            max_tts_words: Maximum words per chunk dispatched to the TTS worker.
            max_buffer_words: Maximum words in buffer before triggering a forced flush.
            on_chunk: Callback invoked as callback(chunk_text, label).
        """
        self.max_tts_words = max(4, max_tts_words)
        self.max_buffer_words = max(self.max_tts_words, max_buffer_words)
        self.on_chunk = on_chunk
        self.buffer = ""
        self.emitted_chunks: List[str] = []

    def _split_into_word_pieces(self, text: str) -> List[str]:
        """Split text into pieces of at most max_tts_words words."""
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        words = cleaned.split()
        return [
            " ".join(words[i : i + self.max_tts_words])
            for i in range(0, len(words), self.max_tts_words)
        ]

    def _buffer_word_count(self) -> int:
        """Count current words residing in the buffer."""
        return len(self.buffer.split()) if self.buffer.strip() else 0

    def _dispatch(self, text: str, label: str) -> None:
        """Partition text into max_tts_words pieces and dispatch each."""
        for piece in self._split_into_word_pieces(text):
            if piece.strip():
                self.emitted_chunks.append(piece)
                if self.on_chunk:
                    self.on_chunk(piece, label)

    def process_token(self, token: str) -> None:
        """
        Append a streamed token and check for sentence boundaries or buffer overflow.
        """
        if not token:
            return

        self.buffer += token

        # Sentence boundary detection: regex lookbehind for [.!?] followed by a space
        sentences = re.split(r"(?<=[.!?]) +", self.buffer)
        
        # The last segment is incomplete and remains in the buffer
        self.buffer = sentences.pop() if sentences else ""

        # Dispatch complete sentences
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                self._dispatch(sentence, "TTS_SENTENCE")

        # Buffer overflow prevention: if text without punctuation exceeds max_buffer_words
        while self._buffer_word_count() > self.max_buffer_words:
            words = self.buffer.split()
            piece = " ".join(words[: self.max_tts_words])
            self.buffer = " ".join(words[self.max_tts_words :])
            self._dispatch(piece, "BUFFER_OVERFLOW")

    def finalize(self) -> None:
        """
        Flush any remaining text in the buffer when the LLM stream signals completion.
        """
        remaining = self.buffer.strip()
        if remaining:
            self._dispatch(remaining, "FINAL_FLUSH")
            self.buffer = ""

