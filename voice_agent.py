"""
Voice Agent for Elyestra Voice Pipeline
=======================================
Implements the core TTS inference engine using Kokoro-82M ONNX Runtime.

Technical Architecture & Evolution:
-----------------------------------
1. Initial Architecture (Autoregressive / Heavy PyTorch):
   - Model: Qwen3TTS / Tortoise / XTTS style architectures.
   - Mechanism: Loaded PyTorch weights and dynamic reference audio prompts (voice_clone_prompt.pt).
   - Bottleneck: High latency on CPU hardware. Acoustic tokens were generated autoregressively
     (token-by-token loop), causing Real-Time Factor (RTF) > 1.5. A 10-word chunk took 4-10 seconds,
     starving the client audio playback queue.
   - Dynamic representations: Reference speaker embeddings had to be recalculated or maintained
     in an embedding cache (hence `clear_embedding_cache()`).

2. Modern Solution (Kokoro-82M Non-Autoregressive ONNX):
   - Model: Kokoro-82M v1.0 (StyleTTS 2 + ISTFTNet architecture).
   - Mechanism: Pre-quantized ONNX graph inference.
   - Pre-computed Style Vectors: Instead of extracting speaker embeddings at runtime, voice profiles
     (e.g., 'af_bella') are stored as pre-computed 1D style vectors in `voices-v1.0.bin`.
   - Zero Runtime Representation Overhead: The static style vector is passed directly into the
     ONNX forward pass. No PyTorch execution, no acoustic token looping, no embedding cache invalidation.
   - Performance: Single-pass parallel waveform generation in ~150-300ms on CPU (RTF < 0.15),
     enabling continuous audio streaming that outpaces speech playback.
"""

import os
import time
from pathlib import Path
from threading import Lock
from typing import Tuple, Dict, Optional
import numpy as np

# Optional imports for local audio file generation
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from kokoro_onnx import Kokoro
    HAS_KOKORO = True
except ImportError:
    HAS_KOKORO = False

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_bella")
KOKORO_LANG = os.getenv("KOKORO_LANG", "en-us")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))


class VoiceAgent:
    """
    Synthesizes sentence chunks into audio waveforms using Kokoro ONNX or high-fidelity simulation.
    """

    def __init__(
        self,
        chats_dir: str = "chats",
        model_path: str = "kokoro-v1.0.onnx",
        voices_path: str = "voices-v1.0.bin",
        mock_mode: bool = False,
    ):
        self.chats_dir = Path(chats_dir)
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self.mock_mode = mock_mode
        self._kokoro = None
        self._chunk_counters: Dict[int, int] = {}
        self._counter_lock = Lock()

        if not self.mock_mode and HAS_KOKORO and os.path.exists(model_path) and os.path.exists(voices_path):
            print(f"⏳ [VoiceAgent] Loading Kokoro ONNX pipeline ({model_path}, {voices_path})...")
            try:
                self._kokoro = Kokoro(model_path, voices_path)
                print(f"✅ [VoiceAgent] Kokoro loaded — active voice: {KOKORO_VOICE}")
            except Exception as e:
                print(f"⚠️ [VoiceAgent] Could not load Kokoro ({e}), falling back to simulation mode.")
                self.mock_mode = True
        else:
            if not self.mock_mode:
                print("ℹ️ [VoiceAgent] ONNX model weights not found locally. Initialized in Simulation Mode.")
            self.mock_mode = True

    def clear_embedding_cache(self) -> None:
        """
        Historical artifact kept for backwards compatibility.
        In the previous PyTorch architecture, speaker embeddings were dynamically computed and
        cached in RAM. In Kokoro ONNX, static pre-computed style vectors from voices-v1.0.bin
        are used directly, rendering embedding cache invalidation unnecessary.
        """
        pass

    def generate_voice(
        self,
        conversation_id: int,
        text: str,
        language: str = "English",
        sequence_order: Optional[int] = None,
    ) -> str:
        """
        Synthesize audio for a single sentence chunk and write to disk.
        """
        # Resolve sequence number
        if sequence_order is not None:
            seq = sequence_order
        else:
            with self._counter_lock:
                self._chunk_counters[conversation_id] = self._chunk_counters.get(conversation_id, 0) + 1
                seq = self._chunk_counters[conversation_id]

        start_t = time.time()
        audio_array, sr = self._synthesize(text)
        elapsed = time.time() - start_t
        duration_s = len(audio_array) / sr if sr > 0 else 0.0

        # Output folder per conversation
        folder = self.chats_dir / f"chat_{conversation_id}"
        folder.mkdir(parents=True, exist_ok=True)

        output_path = folder / f"chunk_{seq:04d}_{int(time.time() * 1000)}.wav"

        if HAS_SOUNDFILE:
            sf.write(str(output_path), audio_array, sr)
        else:
            # Create a placeholder file if soundfile is not installed
            output_path.write_bytes(b"RIFF_MOCK_WAV_HEADER_DATA")

        print(
            f"⚡ [VoiceAgent] Chunk seq={seq:02d} synthesized in {elapsed:.3f}s "
            f"(Audio length: {duration_s:.2f}s, RTF: {elapsed / max(0.001, duration_s):.2f}) "
            f"→ {output_path.name}"
        )
        return str(output_path)

    def _synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Runs either Kokoro ONNX forward pass or simulated realistic synthesis.
        """
        if not self.mock_mode and self._kokoro is not None:
            audio, sr = self._kokoro.create(
                text,
                voice=KOKORO_VOICE,
                speed=KOKORO_SPEED,
                lang=KOKORO_LANG,
            )
            return audio.astype(np.float32), sr

        # Simulation Mode: Real-time factor of ~0.10 - 0.20 on modern CPU
        # Average reading speed ~ 2.5 words/sec -> 15 words is ~6.0s of audio
        words = len(text.split())
        estimated_audio_duration = max(1.0, words / 2.5)
        # Realistic CPU ONNX latency: ~15-20ms per second of audio
        simulated_inference_latency = 0.10 + (estimated_audio_duration * 0.02)
        time.sleep(simulated_inference_latency)

        sr = 24000
        total_samples = int(sr * estimated_audio_duration)
        # Generate clean synthetic tone for demonstration
        t = np.linspace(0, estimated_audio_duration, total_samples, endpoint=False)
        audio = (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return audio, sr

    def merge_audio_chunks(self, conversation_id: int) -> str:
        """
        Sequentially concatenates all chunk WAV files for this conversation
        into final_output.wav for archival or full replays.
        """
        folder = self.chats_dir / f"chat_{conversation_id}"
        if not folder.exists():
            raise RuntimeError(f"No audio directory found for conversation {conversation_id}")

        chunk_files = sorted(folder.glob("chunk_*.wav"))
        if not chunk_files:
            raise RuntimeError(f"No voice chunk files found in {folder}")

        final_path = folder / "final_output.wav"

        if HAS_SOUNDFILE:
            combined = []
            sample_rate = None
            for p in chunk_files:
                data, sr = sf.read(str(p))
                if sample_rate is None:
                    sample_rate = sr
                combined.append(data)

            final_audio = np.concatenate(combined)
            sf.write(str(final_path), final_audio, sample_rate)
        else:
            final_path.write_bytes(b"RIFF_MOCK_MERGED_WAV")

        with self._counter_lock:
            self._chunk_counters.pop(conversation_id, None)

        print(f"🎬 [VoiceAgent] Final merged audio generated: {final_path} ({len(chunk_files)} chunks combined)")
        return str(final_path)

