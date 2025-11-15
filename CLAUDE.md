# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Whisper Web is a self-hosted speech-to-text service powered by OpenAI's Whisper model. Python/FastAPI backend with vanilla HTML/CSS/JS frontend.

## Development Commands

```bash
# Run development server
python app.py

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Docker deployment
docker build -t whisper-web .
docker-compose up -d
docker-compose up -d --profile with-nginx  # With Nginx reverse proxy

# Health checks
curl http://localhost:5000/health
curl http://localhost:5000/api/status
```

## Architecture

### Backend (app.py)
- **Framework**: FastAPI with async/await throughout
- **Server**: Uvicorn ASGI server on port 5000
- **ML Model**: OpenAI Whisper loaded at startup via lifespan context manager
- **GPU Support**: Auto-detects CUDA, enables FP16 and TF32 optimizations

### Frontend (templates/index.html)
- Single-page application with vanilla JavaScript
- Uses Web Audio API for microphone recording
- Fetch API for HTTP requests
- Local Storage for session persistence

### Data Flow
```
Audio Input → MIME Type Validation → Temp File Creation →
Whisper Processing → JSON History Storage → GPU Cache Cleanup → Response
```

### Storage (File-based, no database)
- `uploads/` - Uploaded audio files with timestamp prefixes
- `transcriptions/` - JSON files containing transcription metadata and results
- `recorded_audio/` - Browser-recorded audio files
- `clipboard/` - Shared clipboard items with optional password protection

### Configuration (config.py)
- Environment variable driven via python-dotenv
- Key settings: WHISPER_MODEL (tiny/base/small/medium/large/large-v3), AUTH_PASSWORD, RATE_LIMIT_*, MAX_FILE_SIZE_MB
- DEVICE auto-detects cuda/cpu

## Key Code Patterns

### Security
- SHA256 password hashing (not salted, suitable for single-password auth)
- URL-safe base64 session tokens (32 chars, 24-hour expiration)
- Per-IP rate limiting with in-memory tracking
- Directory traversal protection via filename validation
- CORS set to allow-all (designed for embedding)

### Request Processing
- `get_client_ip()` handles reverse proxy headers (X-Real-IP, X-Forwarded-For)
- `validate_audio_file()` checks MIME type and extension
- `save_transcription()` persists results with client IP, timestamps, audio duration

### GPU Memory Management
- Model loads once at startup in lifespan manager
- `torch.cuda.empty_cache()` called after each transcription
- PyTorch optimizations: cudnn.benchmark=True, matmul.allow_tf32=True

## External Integrations

- **Ollama** (localhost:11434): Text polish feature uses qwen2.5:7b model for punctuation and error correction
- **NVIDIA CUDA**: Optional GPU acceleration with CUDA 12.1
- **FFmpeg**: System dependency for audio format handling

## API Structure

Main endpoints:
- `POST /api/transcribe` - Transcription (multipart form or base64)
- `POST /api/auth` - Password authentication
- `GET /api/history` - Paginated transcription history (requires auth)
- `POST /api/polish` - LLM text enhancement via Ollama
- `/api/clipboard/*` - Clipboard sharing with optional 4-digit passwords
- `GET /health` and `GET /api/status` - Health monitoring
