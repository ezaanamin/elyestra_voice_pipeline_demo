# ELYESTRA — Streaming Voice Architecture

Low-latency, multi-voice text-to-speech pipeline for AI assistants, built to run on CPU-constrained hardware.

## Overview

ELYESTRA streams LLM output sentence-by-sentence into a FIFO TTS queue, so speech synthesis begins before the full response has finished generating. The goal: minimize perceived latency between a prompt and the first audible word, without requiring GPU-class hardware.

This repo documents both the architecture and the reasoning behind a key model swap — from an autoregressive TTS model to a non-autoregressive one — after discovering that the bottleneck wasn't the pipeline, it was the model.

## Architecture

```
CLIENT                CORE PIPELINE              VOICE SERVICE
┌─────────────┐  prompt   ┌───────────┐  token stream  ┌──────────────────┐
│ Client       │ ───────► │ Local LLM │ ─────────────► │ Sentence Chunker  │
│ Text stream  │          └───────────┘                └──────────────────┘
│ Audio playback│                                              │
└─────────────┘                                                ▼
                                                        ┌──────────────────┐
                                                        │ FIFO TTS Queue   │
                                                        │ chunk 01, 02...  │
                                                        └──────────────────┘
                                                                │
                                                                ▼
                                                        ┌──────────────────┐
                                                        │ Kokoro TTS       │
                                                        │ Inference        │
                                                        └──────────────────┘
                                                                │
                                                                ▼
                                                        ┌──────────────────┐   ┌────────────────┐
                                                        │ Client Audio     │ ► │ Audio Playback │
                                                        │ Queue            │   └────────────────┘
                                                        └──────────────────┘
```

**Flow:**
1. Client sends a text prompt to the local LLM.
2. The LLM streams tokens as they're generated.
3. A sentence chunker detects complete sentences in the token stream and pushes each one into a FIFO queue.
4. The TTS engine pulls chunks off the queue in order and synthesizes audio for each one independently.
5. Generated audio is pushed to a client-side audio queue and played back sequentially — so playback can begin on chunk 01 while chunk 04 hasn't been generated yet.

## Why this exists

The initial assumption was that latency could be solved entirely at the application layer: split the voice agent into its own service, stream tokens instead of waiting for the full response, and process TTS in a queue so synthesis overlaps with generation.

That architecture worked exactly as designed — and voice generation was still slow.

The actual bottleneck was one level lower: the TTS model itself.

## The model problem

The original TTS model, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, is **autoregressive** — each output frame depends on the one generated before it:

```
output_1 → output_2 → output_3 → output_4 → ...
```

This sequential dependency means inference time scales with output length and can't be meaningfully parallelized, regardless of how well the surrounding pipeline is architected. On a 12th-gen Intel i5 with no GPU acceleration, this made per-chunk synthesis too slow to hit low-latency targets — no amount of queuing or streaming at the application layer could fix a bottleneck that lived inside the model's inference loop.

**Fix:** switched to **Kokoro**, a non-autoregressive TTS model, which generates output without that sequential dependency and is far better suited to constrained CPU hardware.

## Lesson

You can't treat models as black boxes. Understanding *how* a model performs inference — and how that interacts with your hardware — determines whether the right fix is in your application code or in your model choice. Sometimes the bottleneck isn't the API, the queue, or the microservice architecture. It's the model.

## Stack

- **LLM:** local inference (streaming token output)
- **Sentence chunking:** custom, detects sentence boundaries in a token stream
- **Queue:** FIFO, in-memory, decouples generation speed from synthesis speed
- **TTS engine:** [Kokoro](https://github.com/hexgrad/kokoro) — non-autoregressive, CPU-friendly
- **Deployment:** Docker

## Running it

```bash
docker compose up
```

The voice service runs independently from the core LLM pipeline, communicating over the FIFO queue described above. See `docker-compose.yml` and the `voice-service/` directory for service-specific configuration.

## Reproducing this

The full implementation is Dockerized so the pipeline — including the Kokoro TTS integration — can be reproduced and experimented with independently. If you're hitting similar latency issues with an autoregressive TTS model on CPU-only hardware, this is a working reference for the non-autoregressive swap.

## License

MIT