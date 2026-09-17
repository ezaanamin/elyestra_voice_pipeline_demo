"""
TTS Worker for Elyestra Voice Pipeline
======================================
Asynchronous background worker that decouples the fast LLM streaming token loop
from HTTP network I/O and TTS inference.

Design:
- Uses a FIFO queue.Queue and a daemon Thread.
- Maintains sequence order across chunks.
- Submits chunks to the Voice Service via POST /tts_chunk.
- Triggers audio concatenation via POST /tts_finalize upon completion.
- Supports graceful shutdown with poison-pill signaling.
"""

import asyncio
import queue
import time
from threading import Thread
from typing import Optional, Callable
import requests


class TTSWorker:
    """
    Background worker consuming chunk synthesis requests sequentially.
    """

    def __init__(
        self,
        voice_service_url: str = "http://localhost:8001",
        timeout: float = 60.0,
        on_audio_ready: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self._url = voice_service_url.rstrip("/")
        self._timeout = timeout
        self._q: queue.Queue = queue.Queue()
        self._thread: Thread | None = None
        self._on_audio_ready = on_audio_ready

    def start(self) -> None:
        """Start the background worker thread (no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run, daemon=True, name="TTSWorkerThread")
        self._thread.start()

    async def shutdown(self) -> None:
        """Send the poison pill and wait for the worker thread to finish."""
        self._q.put(None)
        if self._thread is not None and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, self._timeout)

    def shutdown_sync(self) -> None:
        """Synchronous shutdown for non-async environments."""
        self._q.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(self._timeout)

    def enqueue_chunk(self, conv_id: int, text: str, seq: int) -> None:
        """Add a sentence chunk to the queue to be voiced."""
        preview = (text[:80] + "…") if len(text) > 80 else text
        print(f"📥 [TTSWorker Queue] seq={seq:02d} ({len(text.split()):02d} words) → \"{preview}\"")
        self._q.put(("chunk", conv_id, text, seq))

    def enqueue_merge(self, conv_id: int) -> None:
        """Add a merge job to concatenate all chunks after stream ends."""
        print(f"📥 [TTSWorker Queue] Final merge queued for conversation {conv_id}")
        self._q.put(("merge", conv_id))

    def _generate_chunk_audio(self, conv_id: int, text: str, seq: int) -> bool:
        start_t = time.time()
        try:
            r = requests.post(
                f"{self._url}/tts_chunk",
                json={"conversation_id": conv_id, "text": text, "sequence": seq},
                timeout=self._timeout,
            )
            elapsed = time.time() - start_t
            if r.status_code >= 400:
                print(f"❌ [TTSWorker] Error seq={seq}: HTTP {r.status_code} - {r.text[:100]}")
                return False

            res_json = r.json()
            chunk_url = res_json.get("chunk", "")
            print(f"🔊 [TTSWorker] Audio seq={seq:02d} READY in {elapsed:.2f}s → {chunk_url}")
            if self._on_audio_ready:
                self._on_audio_ready(conv_id, seq, chunk_url)
            return True
        except Exception as e:
            print(f"❌ [TTSWorker] Connection error seq={seq}: {e}")
            return False

    def _finalize_audio(self, conv_id: int) -> Optional[str]:
        start_t = time.time()
        try:
            r = requests.post(
                f"{self._url}/tts_finalize",
                json={"conversation_id": conv_id},
                timeout=self._timeout,
            )
            elapsed = time.time() - start_t
            if r.status_code >= 400:
                print(f"❌ [TTSWorker] Merge error: HTTP {r.status_code} - {r.text[:100]}")
                return None
            final_audio = r.json().get("final_audio")
            print(f"🎉 [TTSWorker] All chunks merged in {elapsed:.2f}s → {final_audio}")
            return final_audio
        except Exception as e:
            print(f"❌ [TTSWorker] Merge connection error: {e}")
            return None

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            op, *rest = item
            if op == "chunk":
                conv_id, text, seq = rest
                self._generate_chunk_audio(conv_id, text, seq)
            elif op == "merge":
                self._finalize_audio(rest[0])
            self._q.task_done()

