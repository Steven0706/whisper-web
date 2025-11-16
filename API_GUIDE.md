# Whisper Web API Guide

This document provides detailed API documentation for integrating with the Whisper Web speech-to-text service.

## Base URL

```
http://your-server:5000
```

## Overview

Whisper Web provides three main transcription endpoints:
1. **REST API** - Non-streaming batch transcription
2. **WebSocket API** - Real-time streaming transcription with optional Ollama polishing
3. **Polish API** - Text post-processing with LLM

---

## 1. Non-Streaming Transcription (REST API)

### Endpoint: `POST /api/transcribe`

Batch transcription of complete audio files. Best for pre-recorded audio.

#### Request Format (Form Data)

```http
POST /api/transcribe
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio` | File | No* | - | Audio file upload (WAV, MP3, WebM, OGG, M4A, FLAC) |
| `audio_base64` | String | No* | - | Base64-encoded audio data |
| `filename` | String | No | "audio.wav" | Filename for the audio |
| `language` | String | No | auto-detect | ISO 639-1 language code (e.g., "en", "zh", "ja") |
| `task` | String | No | "transcribe" | "transcribe" or "translate" (to English) |

*Either `audio` or `audio_base64` must be provided.

#### Example: File Upload (cURL)

```bash
curl -X POST "http://localhost:5000/api/transcribe" \
  -F "audio=@recording.wav" \
  -F "language=en" \
  -F "task=transcribe"
```

#### Example: Base64 Audio (cURL)

```bash
curl -X POST "http://localhost:5000/api/transcribe" \
  -F "audio_base64=$(base64 -w0 recording.wav)" \
  -F "filename=recording.wav" \
  -F "language=zh"
```

#### Example: Python

```python
import requests

# File upload
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:5000/api/transcribe",
        files={"audio": f},
        data={"language": "en", "task": "transcribe"}
    )
    result = response.json()
    print(result["text"])

# Base64
import base64
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:5000/api/transcribe",
    data={
        "audio_base64": audio_b64,
        "filename": "audio.wav",
        "language": "en"
    }
)
```

#### Example: JavaScript/Node.js

```javascript
// File upload
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('audio', fs.createReadStream('audio.wav'));
form.append('language', 'en');

const response = await axios.post('http://localhost:5000/api/transcribe', form, {
  headers: form.getHeaders()
});
console.log(response.data.text);
```

#### Success Response (200 OK)

```json
{
  "success": true,
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "processing_time": "2.34s",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": " Hello, this is a test transcription."
    }
  ],
  "device": "cuda",
  "model": "large-v3"
}
```

#### Error Responses

| Status | Description |
|--------|-------------|
| 400 | Invalid request (missing audio, invalid file type, invalid task) |
| 413 | File too large (max 100MB by default) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

### Endpoint: `POST /api/transcribe/json`

Alternative JSON-based endpoint for base64 audio.

#### Request Format

```http
POST /api/transcribe/json
Content-Type: application/json
```

```json
{
  "audio": "BASE64_ENCODED_AUDIO_DATA",
  "filename": "recording.webm",
  "language": "en",
  "task": "transcribe"
}
```

#### Example: Python

```python
import requests
import base64

with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:5000/api/transcribe/json",
    json={
        "audio": audio_b64,
        "filename": "audio.wav",
        "language": "en",
        "task": "transcribe"
    }
)
result = response.json()
```

---

## 2. Streaming Transcription (WebSocket API)

### Endpoint: `ws://your-server:5000/ws/transcribe`

Real-time streaming transcription with overlapping chunks and optional LLM text polishing.

#### Connection Flow

1. **Connect** to WebSocket endpoint
2. **Send configuration** message (JSON)
3. **Stream audio** chunks (binary 16-bit PCM)
4. **Receive** partial transcriptions and polished text
5. **Send end** signal
6. **Receive** final transcription

#### Configuration Message (First Message)

```json
{
  "type": "config",
  "language": "en",
  "task": "transcribe",
  "sample_rate": 16000,
  "enable_polish": true
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `type` | String | Yes | - | Must be "config" |
| `language` | String | No | auto | ISO 639-1 language code |
| `task` | String | No | "transcribe" | "transcribe" or "translate" |
| `sample_rate` | Number | No | 16000 | Audio sample rate in Hz |
| `enable_polish` | Boolean | No | true | Enable real-time Ollama text polishing |

#### Audio Data Format

- **Format**: 16-bit signed PCM (Little Endian)
- **Sample Rate**: 16000 Hz (recommended)
- **Channels**: Mono
- **Chunk Size**: Recommended 4096-8192 bytes per message

#### Server Messages

**Ready Message:**
```json
{
  "type": "ready",
  "message": "Streaming transcription ready",
  "model": "large-v3",
  "device": "cuda",
  "polish_enabled": true
}
```

**Partial Transcription:**
```json
{
  "type": "partial",
  "text": "newly transcribed text",
  "raw_text": "raw chunk transcription",
  "full_transcript": "complete transcription so far",
  "language": "en",
  "processing_time": "0.85s",
  "buffer_duration": "15.23s",
  "total_duration": "10.00s",
  "is_final": false
}
```

**Polished Text (when enable_polish=true):**
```json
{
  "type": "polished",
  "polished_transcript": "Complete transcription with punctuation and corrections."
}
```

**Final Result:**
```json
{
  "type": "final",
  "text": "last chunk text",
  "full_transcript": "Complete final transcription",
  "polished_transcript": "Complete polished transcription with punctuation.",
  "segments": [...],
  "total_duration": "25.50s",
  "is_final": true
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Error description"
}
```

#### End Stream Signal

```json
{
  "type": "end"
}
```

#### Example: Python Client

```python
import asyncio
import websockets
import json
import wave
import struct

async def stream_transcribe(audio_file_path):
    uri = "ws://localhost:5000/ws/transcribe"

    async with websockets.connect(uri) as websocket:
        # Send configuration
        config = {
            "type": "config",
            "language": "en",
            "task": "transcribe",
            "sample_rate": 16000,
            "enable_polish": True
        }
        await websocket.send(json.dumps(config))

        # Wait for ready
        response = await websocket.recv()
        ready = json.loads(response)
        print(f"Server ready: {ready}")

        # Read and stream audio file
        with wave.open(audio_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            chunk_size = 4096  # bytes

            while True:
                frames = wav_file.readframes(chunk_size // 2)  # 2 bytes per sample
                if not frames:
                    break

                # Send binary audio data
                await websocket.send(frames)

                # Check for responses (non-blocking)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    data = json.loads(response)

                    if data["type"] == "partial":
                        print(f"Partial: {data['full_transcript']}")
                    elif data["type"] == "polished":
                        print(f"Polished: {data['polished_transcript']}")
                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(0.1)  # Simulate real-time streaming

        # Signal end of stream
        await websocket.send(json.dumps({"type": "end"}))

        # Receive final result
        while True:
            response = await websocket.recv()
            data = json.loads(response)

            if data["type"] == "final":
                print(f"\nFinal transcript: {data['full_transcript']}")
                if "polished_transcript" in data:
                    print(f"Polished: {data['polished_transcript']}")
                break
            elif data["type"] == "partial":
                print(f"Partial: {data['full_transcript']}")
            elif data["type"] == "polished":
                print(f"Polished: {data['polished_transcript']}")

# Run
asyncio.run(stream_transcribe("audio.wav"))
```

#### Example: JavaScript/Browser

```javascript
class StreamingTranscriber {
  constructor(serverUrl = 'ws://localhost:5000/ws/transcribe') {
    this.serverUrl = serverUrl;
    this.ws = null;
    this.audioContext = null;
    this.mediaStream = null;
    this.processor = null;
  }

  async start(options = {}) {
    const {
      language = null,
      task = 'transcribe',
      enablePolish = true,
      onPartial = () => {},
      onPolished = () => {},
      onFinal = () => {},
      onError = () => {}
    } = options;

    // Connect WebSocket
    this.ws = new WebSocket(this.serverUrl);

    this.ws.onopen = () => {
      // Send configuration
      this.ws.send(JSON.stringify({
        type: 'config',
        language: language,
        task: task,
        sample_rate: 16000,
        enable_polish: enablePolish
      }));
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'ready':
          console.log('Server ready:', data);
          this.startRecording();
          break;
        case 'partial':
          onPartial(data);
          break;
        case 'polished':
          onPolished(data);
          break;
        case 'final':
          onFinal(data);
          break;
        case 'error':
          onError(data);
          break;
      }
    };

    this.ws.onerror = (error) => {
      onError({ message: 'WebSocket error' });
    };
  }

  async startRecording() {
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    this.audioContext = new AudioContext({ sampleRate: 16000 });
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);

    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        const inputData = e.inputBuffer.getChannelData(0);

        // Convert Float32 to Int16
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        this.ws.send(int16Data.buffer);
      }
    };

    source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
  }

  stop() {
    // Stop recording
    if (this.processor) {
      this.processor.disconnect();
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
    }
    if (this.audioContext) {
      this.audioContext.close();
    }

    // Send end signal
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'end' }));
    }
  }
}

// Usage
const transcriber = new StreamingTranscriber();

transcriber.start({
  language: 'en',
  enablePolish: true,
  onPartial: (data) => {
    console.log('Transcript:', data.full_transcript);
  },
  onPolished: (data) => {
    console.log('Polished:', data.polished_transcript);
  },
  onFinal: (data) => {
    console.log('Final:', data.full_transcript);
    console.log('Polished:', data.polished_transcript);
  }
});

// Stop after 10 seconds
setTimeout(() => transcriber.stop(), 10000);
```

#### Streaming Features

- **Overlapping Chunks**: 10-second processing windows with 5-second overlap for continuity
- **Smart Boundaries**: Silence detection for optimal chunk splitting
- **Deduplication**: Automatic removal of repeated words from overlapping segments
- **Real-time Polish**: Optional LLM-based text correction using Ollama (qwen3-coder:30b)
- **Short Input Support**: Handles recordings of any length, including very short ones (<10s)

---

## 3. Text Polishing API

### Endpoint: `POST /api/polish`

Post-process transcribed text to add punctuation and fix errors using Ollama LLM.

#### Request Format

```http
POST /api/polish
Content-Type: application/json
```

```json
{
  "text": "hello this is a test i want to add punctuation to this text"
}
```

#### Example: Python

```python
import requests

response = requests.post(
    "http://localhost:5000/api/polish",
    json={
        "text": "hello this is a test transcription without any punctuation it needs to be fixed"
    }
)
result = response.json()
print(result["polished"])
# Output: "Hello, this is a test transcription without any punctuation. It needs to be fixed."
```

#### Example: cURL

```bash
curl -X POST "http://localhost:5000/api/polish" \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world this is a test"}'
```

#### Success Response (200 OK)

```json
{
  "success": true,
  "original": "hello this is a test i want to add punctuation",
  "polished": "Hello, this is a test. I want to add punctuation.",
  "model": "qwen3-coder:30b"
}
```

#### Error Responses

| Status | Description |
|--------|-------------|
| 400 | No text provided or invalid JSON |
| 500 | LLM service unavailable |
| 504 | LLM service timeout |

---

## 4. Utility Endpoints

### Health Check

```http
GET /health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model": "large-v3"
}
```

### Server Status

```http
GET /api/status
```

```json
{
  "status": "online",
  "device": "cuda",
  "model": "large-v3",
  "cuda_available": true,
  "gpu": {
    "name": "NVIDIA GeForce RTX 5090",
    "memory_allocated": "2.45 GB",
    "memory_reserved": "3.12 GB",
    "memory_total": "32.00 GB"
  },
  "cuda_version": "12.8"
}
```

---

## Rate Limiting

- Default: 30 requests per 60-second window
- Applies to all transcription endpoints
- Returns HTTP 429 when exceeded

## File Size Limits

- Maximum file size: 100MB (configurable)
- Supported formats: WAV, MP3, WebM, OGG, M4A, FLAC

## Language Codes

Common language codes (ISO 639-1):
- `en` - English
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `es` - Spanish
- `fr` - French
- `de` - German
- `ru` - Russian
- `ar` - Arabic
- `hi` - Hindi

Leave `language` as `null` or omit for automatic language detection.

---

## Complete Integration Example

```python
import requests
import asyncio
import websockets
import json
import base64

class WhisperWebClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")

    def transcribe_file(self, file_path, language=None, task="transcribe"):
        """Non-streaming transcription"""
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/api/transcribe",
                files={"audio": f},
                data={"language": language, "task": task}
            )
        return response.json()

    def polish_text(self, text):
        """Polish transcribed text"""
        response = requests.post(
            f"{self.base_url}/api/polish",
            json={"text": text}
        )
        return response.json()

    async def stream_transcribe(self, audio_generator, language=None, enable_polish=True):
        """Streaming transcription"""
        async with websockets.connect(f"{self.ws_url}/ws/transcribe") as ws:
            # Configure
            await ws.send(json.dumps({
                "type": "config",
                "language": language,
                "sample_rate": 16000,
                "enable_polish": enable_polish
            }))

            # Wait for ready
            await ws.recv()

            # Stream audio
            for chunk in audio_generator:
                await ws.send(chunk)

                # Process responses
                try:
                    msg = await asyncio.wait_for(ws.recv(), 0.01)
                    yield json.loads(msg)
                except asyncio.TimeoutError:
                    pass

            # End stream
            await ws.send(json.dumps({"type": "end"}))

            # Get final
            while True:
                msg = json.loads(await ws.recv())
                yield msg
                if msg.get("type") == "final":
                    break

# Usage
client = WhisperWebClient()

# Non-streaming
result = client.transcribe_file("audio.wav", language="en")
print(f"Text: {result['text']}")

# Polish
polished = client.polish_text(result['text'])
print(f"Polished: {polished['polished']}")

# Streaming (async)
async def stream_example():
    def audio_chunks():
        with open("audio.wav", "rb") as f:
            f.read(44)  # Skip WAV header
            while chunk := f.read(4096):
                yield chunk

    async for msg in client.stream_transcribe(audio_chunks()):
        if msg["type"] == "partial":
            print(f"Live: {msg['full_transcript']}")
        elif msg["type"] == "final":
            print(f"Final: {msg['full_transcript']}")

asyncio.run(stream_example())
```

---

## Performance Considerations

1. **Streaming vs Non-Streaming**: Use streaming for real-time feedback; non-streaming for batch processing
2. **GPU Memory**: Large-v3 model uses ~3GB VRAM; streaming uses additional memory for buffers
3. **Polish Overhead**: Ollama polishing adds 1-3 seconds per request (30B model)
4. **Network Latency**: WebSocket streaming benefits from low-latency connections
5. **Audio Quality**: Higher quality audio = better transcription accuracy

## Troubleshooting

- **Connection refused**: Ensure server is running on correct port
- **Rate limit exceeded**: Wait 60 seconds or adjust `RATE_LIMIT_REQUESTS` in `.env`
- **Model not loaded**: Check GPU memory and model availability
- **Polish not working**: Verify Ollama is running (`ollama serve`) with correct model
- **WebSocket timeout**: Send keep-alive pings for long idle periods
