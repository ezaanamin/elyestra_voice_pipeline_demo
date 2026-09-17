"""
Voice Service Microservice for Elyestra Voice Pipeline
======================================================
FastAPI microservice handling audio chunk synthesis, SSE chunk streaming,
and sequential chunk merging.

Key Architectural Safeguards:
1. Global Mutual Exclusion Lock (_tts_lock):
   Ensures only one chunk synthesis or merge executes at any moment, preventing CPU/RAM
   thrashing on constrained hardware.
2. In-Memory Streaming State (ready_chunks, conv_merge_done):
   Decouples chunk production from client consumption via Server-Sent Events (SSE).
3. Explicit Synchronous Endpoints:
   Replacing the older asynchronous in-service queue (`/enqueue_chunk`) with a blocking
   `POST /tts_chunk` endpoint prevents queue backlog accumulation and guarantees predictable
   resource utilization.
"""

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from voice_agent import VoiceAgent

app = FastAPI(title="Elyestra Voice Service Demo", version="2.0")
agent = VoiceAgent(chats_dir="chats")

# Global lock serializing TTS synthesis and merge jobs
_tts_lock = Lock()

# Per-conversation state for Server-Sent Events (SSE)
chunks_lock = Lock()
ready_chunks: Dict[int, List[dict]] = {}
conv_merge_done: Dict[int, bool] = {}
conv_final_path: Dict[int, str] = {}


def _init_conv_streaming(conv_id: int):
    with chunks_lock:
        ready_chunks.setdefault(conv_id, [])
        conv_merge_done.setdefault(conv_id, False)


def _clear_conv_streaming(conv_id: int):
    with chunks_lock:
        ready_chunks.pop(conv_id, None)
        conv_merge_done.pop(conv_id, None)
        conv_final_path.pop(conv_id, None)


@app.post("/tts_chunk")
def tts_chunk(data: dict):
    """
    Generate one audio chunk synchronously.
    Blocks until the WAV file is synthesized and saved to disk.
    """
    conv_id = int(data.get("conversation_id", 1))
    text = (data.get("text") or "").strip()
    sequence = int(data.get("sequence", 0))

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    with _tts_lock:
        _init_conv_streaming(conv_id)
        try:
            chunk_path = agent.generate_voice(
                conversation_id=conv_id,
                text=text,
                sequence_order=sequence,
            )
            with chunks_lock:
                ready_chunks.setdefault(conv_id, []).append(
                    {"path": chunk_path, "sequence": sequence}
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "ok",
        "sequence": sequence,
        "chunk": f"/chats/chat_{conv_id}/{Path(chunk_path).name}",
    }


@app.post("/tts_finalize")
def tts_finalize(data: dict):
    """
    Concatenate all chunks for a conversation and mark SSE stream as finished.
    """
    conv_id = int(data.get("conversation_id", 1))

    with _tts_lock:
        try:
            merged_path = agent.merge_audio_chunks(conv_id)
            with chunks_lock:
                conv_final_path[conv_id] = merged_path
                conv_merge_done[conv_id] = True
        except Exception as e:
            with chunks_lock:
                conv_merge_done[conv_id] = True
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "merged",
        "final_audio": f"/chats/chat_{conv_id}/{Path(merged_path).name}",
    }


@app.get("/chunks/{conversation_id}")
def stream_chunks(conversation_id: int, timeout: int = 120):
    """
    Server-Sent Events (SSE) endpoint: Streams audio URLs to the frontend
    the moment each chunk completes synthesis.
    """
    conv_id = conversation_id

    def event_stream():
        sent_index = 0
        deadline = time.time() + timeout

        while time.time() < deadline:
            with chunks_lock:
                rows = list(ready_chunks.get(conv_id, []))
                merge_done = conv_merge_done.get(conv_id, False)
                final_abs = conv_final_path.get(conv_id)

            # Emit newly ready chunks
            while sent_index < len(rows):
                row = rows[sent_index]
                sent_index += 1
                payload = json.dumps({
                    "chunk": f"/chats/chat_{conv_id}/{Path(row['path']).name}",
                    "sequence": row["sequence"]
                })
                yield f"data: {payload}\n\n"

            # Check if all chunks and final merge are done
            if merge_done:
                final_url = f"/chats/chat_{conv_id}/{Path(final_abs).name}" if final_abs else "final_output.wav"
                done_payload = json.dumps({"done": True, "final_audio": final_url})
                yield f"data: {done_payload}\n\n"
                _clear_conv_streaming(conv_id)
                return

            time.sleep(0.05)

        yield f"data: {json.dumps({'done': True, 'timeout': True})}\n\n"
        _clear_conv_streaming(conv_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Deprecated Endpoints (Documentation of Architecture Evolution) ───────────

@app.post("/enqueue_chunk")
def deprecated_enqueue_chunk():
    """
    Historical note: The internal queue in the voice service was deprecated because
    unbounded concurrent requests caused CPU starvation and lock contention.
    Queueing was moved client-side to TTSWorker.
    """
    raise HTTPException(
        status_code=410,
        detail="Queue removed — use POST /tts_chunk (blocks until WAV is ready).",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_service:app", host="0.0.0.0", port=8001, reload=False)

