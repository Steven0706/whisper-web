#!/usr/bin/env python3
"""
Whisper Web - Self-hosted Speech Recognition Service
High-performance voice-to-text transcription using OpenAI's Whisper
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from config import Config
import whisper
import torch
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path
import json
import time
import gc
import io
import base64
import numpy as np
import asyncio
from typing import Optional, List, Dict, Any
import uvicorn
import secrets
import hashlib
from datetime import timedelta
import mimetypes
import httpx
import uuid
import soundfile as sf
import wave
import struct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize configuration
Config.init_dirs()

# Global variables for model
model = None
DEVICE = "cuda" if torch.cuda.is_available() and Config.DEVICE != "cpu" else "cpu"
MODEL_NAME = Config.WHISPER_MODEL

# Session storage (in production, use Redis or database)
sessions = {}

def create_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    """Hash password with SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_session(token: str) -> bool:
    """Verify if session token is valid"""
    if token in sessions:
        created_at = sessions[token]
        if datetime.now() - created_at < timedelta(hours=Config.SESSION_DURATION_HOURS):
            return True
        else:
            del sessions[token]
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    global model, MODEL_NAME
    
    logger.info(f"Starting Whisper Web service...")
    logger.info(f"Using device: {DEVICE}")
    
    # Configure PyTorch for optimal performance
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load Whisper model
    try:
        logger.info(f"Loading Whisper model: {MODEL_NAME}")
        model = whisper.load_model(MODEL_NAME, device=DEVICE)
        logger.info(f"Model {MODEL_NAME} loaded successfully on {DEVICE}")
    except Exception as e:
        logger.error(f"Failed to load model {MODEL_NAME}: {e}")
        logger.info("Falling back to base model...")
        MODEL_NAME = "base"
        model = whisper.load_model(MODEL_NAME, device=DEVICE)
        logger.info(f"Loaded fallback model: {MODEL_NAME}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Whisper Web service...")
    if model:
        del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

# Initialize FastAPI
app = FastAPI(
    title="Whisper Web API",
    description="Self-hosted speech-to-text transcription service",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "ALLOWALL"
        response.headers["Content-Security-Policy"] = "frame-ancestors *;"
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"
        response.headers["Cross-Origin-Opener-Policy"] = "unsafe-none"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting storage
rate_limit_storage = {}

def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting check"""
    current_time = time.time()
    
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    
    # Clean old requests
    rate_limit_storage[client_ip] = [
        t for t in rate_limit_storage[client_ip] 
        if current_time - t < Config.RATE_LIMIT_WINDOW
    ]
    
    # Check if limit exceeded
    if len(rate_limit_storage[client_ip]) >= Config.RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limit_storage[client_ip].append(current_time)
    return True

def validate_audio_file(content_type: str, filename: str) -> bool:
    """Validate if file is an allowed audio type"""
    if content_type in Config.ALLOWED_AUDIO_TYPES:
        return True

    # Check by file extension as fallback
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type in Config.ALLOWED_AUDIO_TYPES if mime_type else False

def get_client_ip(request: Request) -> str:
    """Extract real client IP from request headers (considering reverse proxy)"""
    # Try X-Real-IP first (set by Nginx)
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip

    # Try X-Forwarded-For (may contain multiple IPs)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(',')[0].strip()

    # Fallback to direct connection IP
    return request.client.host if request.client else "unknown"

def save_transcription(text: str, filename: str, language: str, processing_time: str, segments: list = None, timestamp: str = None, client_ip: str = None):
    """Save transcription to history"""
    # Use provided timestamp or generate new one
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate audio duration from segments
    audio_duration = 0
    if segments and len(segments) > 0:
        audio_duration = segments[-1].get('end', 0)

    transcription_data = {
        "timestamp": timestamp,
        "filename": filename,
        "text": text[:500],  # Store preview
        "full_text": text,
        "language": language,
        "processing_time": processing_time,
        "model": MODEL_NAME,
        "segments": segments[:10] if segments else [],  # Store first 10 segments
        "audio_file": f"{timestamp}_{filename}",
        "audio_duration": round(audio_duration, 2),  # Duration in seconds
        "client_ip": client_ip or "unknown"  # Store client IP
    }

    json_path = Config.TRANSCRIPTION_DIR / f"{timestamp}_transcription.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_data, f, ensure_ascii=False, indent=2)

    return timestamp

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web interface"""
    template_path = Config.TEMPLATE_DIR / "index.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "model": MODEL_NAME
    }

@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to check all registered routes including WebSocket"""
    routes = []
    for route in app.routes:
        route_info = {
            "path": getattr(route, "path", str(route)),
            "name": getattr(route, "name", "unknown"),
        }
        if hasattr(route, "methods"):
            route_info["methods"] = list(route.methods)
        if "websocket" in str(type(route)).lower():
            route_info["type"] = "websocket"
        else:
            route_info["type"] = "http"
        routes.append(route_info)

    ws_routes = [r for r in routes if r.get("type") == "websocket"]
    return {
        "total_routes": len(routes),
        "websocket_routes": ws_routes,
        "all_routes": routes
    }

@app.get("/debug/headers")
async def debug_headers(request: Request):
    """Debug endpoint to check request headers and IP information"""
    return {
        "remote_addr": request.client.host if request.client else "Unknown",
        "remote_port": request.client.port if request.client else "Unknown",
        "x_real_ip": request.headers.get('X-Real-IP', 'Not found'),
        "x_forwarded_for": request.headers.get('X-Forwarded-For', 'Not found'),
        "x_forwarded_proto": request.headers.get('X-Forwarded-Proto', 'Not found'),
        "forwarded": request.headers.get('Forwarded', 'Not found'),
        "all_headers": dict(request.headers)
    }

@app.get("/api/status")
async def get_status():
    """Get server status and GPU information"""
    status = {
        "status": "online",
        "device": DEVICE,
        "model": MODEL_NAME,
        "cuda_available": torch.cuda.is_available()
    }
    
    if DEVICE == "cuda" and torch.cuda.is_available():
        status["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        }
        status["cuda_version"] = torch.version.cuda
    
    return status

@app.post("/api/transcribe")
async def transcribe_audio(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    filename: Optional[str] = Form("audio.wav"),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe")
) -> Dict[str, Any]:
    """Transcribe audio from file upload or base64 data"""
    
    # Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment.")
    
    # Validate task
    if task not in ["transcribe", "translate"]:
        raise HTTPException(status_code=400, detail="Invalid task. Must be 'transcribe' or 'translate'")
    
    try:
        audio_data = None
        actual_filename = filename
        
        # Handle file upload
        if audio:
            # Validate file type
            if not validate_audio_file(audio.content_type, audio.filename):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
            
            # Check file size
            contents = await audio.read()
            if len(contents) > Config.MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {Config.MAX_FILE_SIZE_MB}MB")
            
            audio_data = contents
            actual_filename = audio.filename
        
        # Handle base64 audio
        elif audio_base64:
            try:
                # Remove data URL prefix if present
                if ',' in audio_base64:
                    audio_base64 = audio_base64.split(',')[1]
                audio_data = base64.b64decode(audio_base64)
                
                if len(audio_data) > Config.MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {Config.MAX_FILE_SIZE_MB}MB")
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        else:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # Save audio temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{actual_filename}"
        
        # Save to permanent storage
        permanent_path = Config.RECORDED_AUDIO_DIR / safe_filename
        with open(permanent_path, 'wb') as f:
            f.write(audio_data)
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=Path(actual_filename).suffix, delete=False) as tmp_file:
            tmp_file.write(audio_data)
            temp_path = tmp_file.name
        
        # Process with Whisper
        start_time = time.time()
        
        # Transcribe with GPU optimization
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                result = model.transcribe(
                    temp_path,
                    language=language,
                    task=task,
                    fp16=(DEVICE == "cuda"),
                    verbose=False
                )
        else:
            result = model.transcribe(
                temp_path,
                language=language,
                task=task,
                verbose=False
            )
        
        processing_time = f"{time.time() - start_time:.2f}s"

        # Get client IP
        client_ip = get_client_ip(request)

        # Save transcription with the same timestamp used for audio file
        save_transcription(
            result["text"],
            actual_filename,
            result.get("language", "unknown"),
            processing_time,
            result.get("segments", []),
            timestamp,  # Pass the timestamp used for audio file
            client_ip   # Pass the client IP
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Clear GPU cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "success": True,
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "processing_time": processing_time,
            "segments": result.get("segments", []),
            "device": DEVICE,
            "model": MODEL_NAME
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe/json")
async def transcribe_audio_json(request: Request) -> Dict[str, Any]:
    """Transcribe audio from JSON request with base64 data"""
    
    # Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment.")
    
    try:
        data = await request.json()
        
        audio_base64 = data.get("audio")
        filename = data.get("filename", "recording.webm")
        language = data.get("language")
        task = data.get("task", "transcribe")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # Forward to main transcribe endpoint
        return await transcribe_audio(
            request=request,
            audio=None,
            audio_base64=audio_base64,
            filename=filename,
            language=language,
            task=task
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JSON transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth")
async def authenticate(password: str = Form(...)) -> Dict[str, Any]:
    """Authenticate user for accessing transcription history"""
    if hash_password(password) == hash_password(Config.AUTH_PASSWORD):
        token = create_session_token()
        sessions[token] = datetime.now()
        return {"success": True, "token": token}
    else:
        raise HTTPException(status_code=401, detail="Invalid password")

@app.post("/api/verify-session")
async def verify_session_endpoint(token: str = Form(...)) -> Dict[str, Any]:
    """Verify if a session token is still valid"""
    return {"valid": verify_session(token)}

@app.get("/api/history")
async def get_history(
    limit: int = 50,
    offset: int = 0,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """Get recent transcription history with pagination (requires authentication)"""

    # Check authentication
    if not token or not verify_session(token):
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        all_files = sorted(Config.TRANSCRIPTION_DIR.glob("*.json"), reverse=True)
        total_count = len(all_files)

        # Collect unique IPs from all files (for filter options)
        unique_ips = set()
        for file in all_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    ip = data.get('client_ip', 'unknown')
                    unique_ips.add(ip)
            except:
                pass

        # Apply pagination
        files = all_files[offset:offset + limit]

        history = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Parse datetime from timestamp if available
                if 'timestamp' in data:
                    try:
                        ts = data['timestamp']
                        parsed_date = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                        data['parsed_datetime'] = parsed_date.isoformat()
                        data['date'] = parsed_date.strftime("%Y-%m-%d")
                        data['time'] = parsed_date.strftime("%H:%M:%S")
                    except:
                        pass

                # Ensure audio_duration exists (for old records)
                if 'audio_duration' not in data:
                    data['audio_duration'] = 0

                # Ensure client_ip exists (for old records)
                if 'client_ip' not in data:
                    data['client_ip'] = 'unknown'

                history.append(data)

        return {
            "items": history,
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total_count,
            "unique_ips": sorted(list(unique_ips))  # Return sorted list of unique IPs
        }
    except Exception as e:
        logger.error(f"Error loading history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load history")

@app.head("/api/audio/{filename}")
@app.get("/api/audio/{filename}")
async def get_audio(filename: str, token: Optional[str] = None):
    """Serve recorded audio files (requires authentication via query param)"""

    # Check authentication - accept token from query param
    if not token or not verify_session(token):
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Security: prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        audio_path = Config.RECORDED_AUDIO_DIR / filename

        # If exact file doesn't exist, try to find a close match (for old records with timestamp mismatch)
        if not audio_path.exists():
            # Extract timestamp from filename (format: YYYYMMDD_HHMMSS_recording.webm)
            try:
                parts = filename.split('_')
                if len(parts) >= 3:
                    date_part = parts[0]  # YYYYMMDD
                    time_part = parts[1]  # HHMMSS
                    suffix = Path(filename).suffix

                    # Try to find files within ±10 seconds
                    base_time = int(time_part)

                    for offset in range(-10, 11):
                        # Calculate new time
                        hours = base_time // 10000
                        minutes = (base_time % 10000) // 100
                        seconds = (base_time % 100) + offset

                        # Handle second overflow/underflow
                        if seconds >= 60:
                            seconds -= 60
                            minutes += 1
                        elif seconds < 0:
                            seconds += 60
                            minutes -= 1

                        # Handle minute overflow/underflow
                        if minutes >= 60:
                            minutes -= 60
                            hours += 1
                        elif minutes < 0:
                            minutes += 60
                            hours -= 1

                        # Skip if hours out of range
                        if hours < 0 or hours >= 24:
                            continue

                        # Build new filename
                        new_time = f"{hours:02d}{minutes:02d}{seconds:02d}"
                        new_filename = f"{date_part}_{new_time}_recording{suffix}"
                        new_path = Config.RECORDED_AUDIO_DIR / new_filename

                        if new_path.exists():
                            logger.info(f"Found matching audio file: {new_filename} for requested {filename}")
                            audio_path = new_path
                            filename = new_filename
                            break
            except Exception as e:
                logger.warning(f"Failed to search for alternative audio file: {e}")

        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

        # Determine media type
        suffix = audio_path.suffix.lower()
        media_type_map = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.webm': 'audio/webm',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            '.flac': 'audio/flac'
        }
        media_type = media_type_map.get(suffix, 'audio/mpeg')

        return FileResponse(
            path=str(audio_path),
            media_type=media_type,
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve audio file")

@app.post("/api/polish")
async def polish_text(request: Request) -> Dict[str, Any]:
    """Polish transcribed text using LLM to add punctuation and fix errors"""
    
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Prepare the prompt
        prompt = f"""你是一个专业的文本编辑助手。下面是一段通过语音识别转换的文字，可能缺少标点符号、段落分隔，并且可能有一些识别错误的词汇。

请帮我：
1. 添加适当的标点符号（句号、逗号、问号、感叹号等）
2. 分段（在适当的地方换行分段）
3. 修正明显的语音识别错误
4. 保持原意不变，只做格式和错误修正

原始文本：
{text}

请直接输出修正后的文本，不要包含任何解释或说明。"""
        
        # Call Ollama API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Lower temperature for more consistent output
                    "top_p": 0.9,
                    "max_tokens": len(text) * 2  # Allow for some expansion
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                raise HTTPException(status_code=500, detail="LLM service unavailable")
            
            result = response.json()
            polished_text = result.get("response", "").strip()
            
            # If no response or error, return original
            if not polished_text:
                polished_text = text
            
            return {
                "success": True,
                "original": text,
                "polished": polished_text,
                "model": "qwen2.5:7b"
            }
            
    except httpx.TimeoutException:
        logger.error("Ollama API timeout")
        raise HTTPException(status_code=504, detail="LLM service timeout")
    except Exception as e:
        logger.error(f"Polish text error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Clipboard endpoints
def get_clipboard_items(limit: int = 10) -> List[Dict[str, Any]]:
    """Get most recent clipboard items"""
    try:
        all_files = sorted(Config.CLIPBOARD_DIR.glob("*.json"), reverse=True)
        items = []

        for file in all_files[:limit]:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Create preview from first 10 characters
                content = data.get('content', '')
                data['preview'] = content[:10] if content else ''

                # Don't include full content in list
                if data.get('password_hash'):
                    data['is_locked'] = True
                else:
                    data['is_locked'] = False

                # Remove full content from list response
                del data['content']
                items.append(data)

        return items
    except Exception as e:
        logger.error(f"Error loading clipboard items: {str(e)}")
        return []

@app.post("/api/clipboard")
async def create_clipboard(request: Request) -> Dict[str, Any]:
    """Create a new clipboard item"""
    try:
        data = await request.json()
        content = data.get("content", "")
        password = data.get("password")  # 4-digit password, optional

        if not content:
            raise HTTPException(status_code=400, detail="Content cannot be empty")

        # Validate password if provided
        if password:
            if not password.isdigit() or len(password) != 4:
                raise HTTPException(status_code=400, detail="Password must be exactly 4 digits")

        # Generate unique ID
        item_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create clipboard item
        item = {
            "id": item_id,
            "content": content,
            "password_hash": hash_password(password) if password else None,
            "created_at": timestamp,
            "updated_at": timestamp
        }

        # Save to file
        filename = f"{timestamp}_{item_id}.json"
        filepath = Config.CLIPBOARD_DIR / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

        logger.info(f"Created clipboard item: {item_id}")

        return {
            "success": True,
            "id": item_id,
            "created_at": timestamp
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create clipboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clipboard")
async def list_clipboard() -> Dict[str, Any]:
    """Get list of clipboard items (latest 10)"""
    try:
        items = get_clipboard_items(limit=10)
        return {
            "success": True,
            "items": items,
            "count": len(items)
        }
    except Exception as e:
        logger.error(f"List clipboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clipboard/{item_id}/verify")
async def verify_clipboard_password(item_id: str, request: Request) -> Dict[str, Any]:
    """Verify password and get clipboard content"""
    try:
        data = await request.json()
        password = data.get("password", "")

        # Find the clipboard item
        files = list(Config.CLIPBOARD_DIR.glob(f"*_{item_id}.json"))

        if not files:
            raise HTTPException(status_code=404, detail="Clipboard item not found")

        filepath = files[0]

        with open(filepath, 'r', encoding='utf-8') as f:
            item = json.load(f)

        # Check if password protected
        if not item.get('password_hash'):
            # No password, return content directly
            return {
                "success": True,
                "content": item['content'],
                "is_locked": False
            }

        # Verify password
        if hash_password(password) != item['password_hash']:
            raise HTTPException(status_code=401, detail="Incorrect password")

        return {
            "success": True,
            "content": item['content'],
            "is_locked": True
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify clipboard password error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clipboard/{item_id}")
async def get_clipboard_item(item_id: str) -> Dict[str, Any]:
    """Get clipboard item (only if no password)"""
    try:
        # Find the clipboard item
        files = list(Config.CLIPBOARD_DIR.glob(f"*_{item_id}.json"))

        if not files:
            raise HTTPException(status_code=404, detail="Clipboard item not found")

        filepath = files[0]

        with open(filepath, 'r', encoding='utf-8') as f:
            item = json.load(f)

        # Check if password protected
        if item.get('password_hash'):
            raise HTTPException(status_code=403, detail="Password required")

        return {
            "success": True,
            "content": item['content'],
            "is_locked": False
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get clipboard item error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/clipboard/{item_id}")
async def delete_clipboard_item(item_id: str) -> Dict[str, Any]:
    """Delete a clipboard item"""
    try:
        # Find the clipboard item
        files = list(Config.CLIPBOARD_DIR.glob(f"*_{item_id}.json"))

        if not files:
            raise HTTPException(status_code=404, detail="Clipboard item not found")

        filepath = files[0]
        os.unlink(filepath)

        logger.info(f"Deleted clipboard item: {item_id}")

        return {
            "success": True,
            "message": "Clipboard item deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete clipboard item error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Streaming Transcription
class StreamingTranscriber:
    """Handle streaming audio transcription with overlapping chunks and smart deduplication"""

    def __init__(self, language: Optional[str] = None, task: str = "transcribe"):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.language = language
        self.task = task
        self.full_transcript = ""
        self.polished_transcript = ""
        self.segments = []
        self.processed_audio_position = 0  # Position in samples of last processed end

        # Overlapping chunk configuration
        self.chunk_duration = 10.0  # Process 10 seconds at a time
        self.overlap_duration = 5.0  # Keep 5 seconds overlap
        self.min_new_audio = 5.0  # Process when we have 5 seconds of new audio

        # For deduplication
        self.last_chunk_text = ""
        self.last_chunk_end_words = []

        # For real-time polishing
        self.unpolished_buffer = ""
        self.polish_enabled = True

        self.last_process_time = time.time()
        self.total_audio_duration = 0.0

    def add_audio_chunk(self, audio_data: bytes, input_sample_rate: int = 16000):
        """Add audio chunk to buffer"""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample if necessary
            if input_sample_rate != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio_array) * self.sample_rate / input_sample_rate)
                audio_array = signal.resample(audio_array, num_samples)

            # Append to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])

        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")
            raise

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds"""
        return len(self.audio_buffer) / self.sample_rate

    def get_new_audio_duration(self) -> float:
        """Get duration of unprocessed audio"""
        unprocessed_samples = len(self.audio_buffer) - self.processed_audio_position
        return max(0, unprocessed_samples / self.sample_rate)

    def should_process(self) -> bool:
        """Check if we have enough new audio to process"""
        return self.get_new_audio_duration() >= self.min_new_audio

    def detect_silence(self, audio: np.ndarray, threshold: float = 0.01) -> List[int]:
        """Detect silence positions in audio for smart boundary detection"""
        window_size = int(self.sample_rate * 0.05)  # 50ms windows
        silence_positions = []

        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i:i + window_size]
            energy = np.sqrt(np.mean(window ** 2))
            if energy < threshold:
                silence_positions.append(i + window_size // 2)

        return silence_positions

    def find_best_split_point(self, target_position: int) -> int:
        """Find the best position to split audio (preferably at silence)"""
        search_range = int(self.sample_rate * 1.0)  # Search within 1 second
        start = max(0, target_position - search_range)
        end = min(len(self.audio_buffer), target_position + search_range)

        if end <= start:
            return target_position

        # Find silence positions in the search range
        search_audio = self.audio_buffer[start:end]
        silence_positions = self.detect_silence(search_audio)

        if silence_positions:
            # Find the silence position closest to target
            adjusted_positions = [pos + start for pos in silence_positions]
            best_pos = min(adjusted_positions, key=lambda x: abs(x - target_position))
            return best_pos

        return target_position

    def deduplicate_text(self, new_text: str) -> str:
        """Remove overlapping text from new transcription"""
        if not self.last_chunk_text or not new_text:
            return new_text

        # Split into words
        new_words = new_text.split()
        last_words = self.last_chunk_text.split()

        if not new_words or not last_words:
            return new_text

        # Find overlap by checking if beginning of new text matches end of last text
        max_overlap = min(len(new_words), len(last_words), 30)  # Check up to 30 words

        best_overlap = 0
        for overlap_size in range(1, max_overlap + 1):
            # Check if last N words of previous chunk match first N words of new chunk
            if last_words[-overlap_size:] == new_words[:overlap_size]:
                best_overlap = overlap_size

        # If we found overlap, remove the overlapping words from new text
        if best_overlap > 0:
            deduplicated = " ".join(new_words[best_overlap:])
            logger.info(f"Removed {best_overlap} overlapping words")
            return deduplicated

        return new_text

    async def polish_text_async(self, text: str) -> str:
        """Polish text using Ollama in real-time"""
        if not text or not self.polish_enabled:
            return text

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:  # Increased timeout for larger model
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5:7b",
                        "prompt": f"""Fix this speech-to-text output. Add punctuation, fix errors. Output ONLY the fixed text, no explanation.

{text}

Fixed:""",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": min(len(text) * 2, 2000)
                        }
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    polished = result.get("response", text).strip()
                    polished = polished.strip('"\'')
                    return polished
        except Exception as e:
            logger.warning(f"Polish failed: {e}")

        return text

    def process_buffer(self) -> Dict[str, Any]:
        """Process current audio buffer with Whisper using overlapping chunks"""
        if len(self.audio_buffer) == 0:
            return {"text": "", "is_final": False}

        start_time = time.time()

        try:
            # Determine what audio to process
            chunk_samples = int(self.chunk_duration * self.sample_rate)
            overlap_samples = int(self.overlap_duration * self.sample_rate)

            # Calculate start position for this chunk (with overlap from last processed)
            if self.processed_audio_position == 0:
                # First chunk, process from beginning
                start_pos = 0
            else:
                # Start from overlap position (go back from last processed position)
                start_pos = max(0, self.processed_audio_position - overlap_samples)
                # Try to find a good split point at silence
                start_pos = self.find_best_split_point(start_pos)

            # Get audio chunk
            audio_to_process = self.audio_buffer[start_pos:]

            # If chunk is too long, truncate it
            if len(audio_to_process) > chunk_samples:
                audio_to_process = audio_to_process[:chunk_samples]

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_to_process, self.sample_rate)
                temp_path = tmp_file.name

            # Transcribe with GPU optimization
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    result = model.transcribe(
                        temp_path,
                        language=self.language,
                        task=self.task,
                        fp16=(DEVICE == "cuda"),
                        verbose=False,
                        condition_on_previous_text=True,
                        initial_prompt=self.full_transcript[-500:] if self.full_transcript else None
                    )
            else:
                result = model.transcribe(
                    temp_path,
                    language=self.language,
                    task=self.task,
                    verbose=False,
                    condition_on_previous_text=True,
                    initial_prompt=self.full_transcript[-500:] if self.full_transcript else None
                )

            # Clean up temp file
            os.unlink(temp_path)

            # Get new text and deduplicate
            raw_text = result["text"].strip()
            new_text = self.deduplicate_text(raw_text)

            # Update last chunk info for next deduplication
            self.last_chunk_text = raw_text

            # Update full transcript
            if new_text:
                if self.full_transcript:
                    self.full_transcript += " " + new_text
                else:
                    self.full_transcript = new_text

            # Update processed position (move forward, but keep overlap for next)
            new_processed_pos = start_pos + len(audio_to_process)
            self.processed_audio_position = new_processed_pos
            self.total_audio_duration = new_processed_pos / self.sample_rate

            # Update segments
            chunk_start_time = start_pos / self.sample_rate
            for seg in result.get("segments", []):
                adjusted_seg = {
                    "start": seg["start"] + chunk_start_time,
                    "end": seg["end"] + chunk_start_time,
                    "text": seg["text"]
                }
                self.segments.append(adjusted_seg)

            processing_time = time.time() - start_time
            self.last_process_time = time.time()

            # Clear GPU cache
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            return {
                "text": new_text,
                "raw_text": raw_text,
                "full_transcript": self.full_transcript,
                "language": result.get("language", "unknown"),
                "processing_time": f"{processing_time:.2f}s",
                "buffer_duration": f"{self.get_buffer_duration():.2f}s",
                "total_duration": f"{self.total_audio_duration:.2f}s",
                "is_final": False
            }

        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
            raise

    def finalize(self) -> Dict[str, Any]:
        """Process any remaining audio and return final result"""
        result = {"text": "", "is_final": True}

        # Process all remaining unprocessed audio, even if it's short
        remaining_duration = self.get_new_audio_duration()

        if remaining_duration > 0.1:  # Process if more than 0.1s remaining (lowered threshold)
            logger.info(f"Finalizing with {remaining_duration:.2f}s of unprocessed audio")

            # Force process by temporarily lowering the threshold
            original_min = self.min_new_audio
            self.min_new_audio = 0.1  # Lower threshold for final processing

            try:
                # Process remaining audio
                if self.get_new_audio_duration() > 0.1:
                    result = self.process_buffer()
            finally:
                self.min_new_audio = original_min

        # If we still haven't processed anything (very short audio), force process entire buffer
        if not self.full_transcript and len(self.audio_buffer) > 0:
            logger.info(f"Processing entire short buffer: {len(self.audio_buffer) / self.sample_rate:.2f}s")
            # Process whatever we have
            original_min = self.min_new_audio
            original_chunk = self.chunk_duration
            self.min_new_audio = 0.0
            self.chunk_duration = len(self.audio_buffer) / self.sample_rate + 1.0  # Make chunk bigger than buffer

            try:
                result = self.process_buffer()
            finally:
                self.min_new_audio = original_min
                self.chunk_duration = original_chunk

        result["is_final"] = True
        result["full_transcript"] = self.full_transcript
        result["segments"] = self.segments
        result["total_duration"] = f"{self.total_audio_duration:.2f}s"

        return result


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for streaming transcription"""
    transcriber = None
    client_ip = "unknown"

    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"WebSocket accept failed: {e}")
        return

    try:
        # Get client IP
        if websocket.client:
            client_ip = websocket.client.host

        # Check rate limit
        if not check_rate_limit(client_ip):
            await websocket.send_json({
                "type": "error",
                "message": "Rate limit exceeded. Please wait a moment."
            })
            await websocket.close()
            return

        logger.info(f"WebSocket connection established from {client_ip}")

        # Wait for configuration message with timeout
        try:
            config_data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"WebSocket config timeout from {client_ip}")
            await websocket.send_json({
                "type": "error",
                "message": "Configuration timeout. Please send config message."
            })
            await websocket.close()
            return
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON config from {client_ip}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": "Invalid JSON configuration"
            })
            await websocket.close()
            return
        except Exception as e:
            logger.error(f"Config receive error from {client_ip}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Configuration error: {str(e)}"
            })
            await websocket.close()
            return

        if config_data.get("type") != "config":
            await websocket.send_json({
                "type": "error",
                "message": "First message must be configuration"
            })
            await websocket.close()
            return

        # Extract configuration
        language = config_data.get("language")
        task = config_data.get("task", "transcribe")
        sample_rate = config_data.get("sample_rate", 16000)
        enable_polish = config_data.get("enable_polish", True)

        if task not in ["transcribe", "translate"]:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid task. Must be 'transcribe' or 'translate'"
            })
            await websocket.close()
            return

        # Initialize transcriber
        transcriber = StreamingTranscriber(language=language, task=task)
        transcriber.polish_enabled = enable_polish

        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "message": "Streaming transcription ready",
            "model": MODEL_NAME,
            "device": DEVICE,
            "polish_enabled": enable_polish
        })

        # Audio collection for saving
        all_audio_chunks = []
        last_polished_length = 0
        connection_closed = False

        # Process incoming audio
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=60.0)

                # Check for disconnect message
                if message.get("type") == "websocket.disconnect":
                    logger.info("WebSocket disconnected by client")
                    connection_closed = True
                    break

                if "bytes" in message:
                    # Binary audio data
                    audio_chunk = message["bytes"]
                    all_audio_chunks.append(audio_chunk)
                    transcriber.add_audio_chunk(audio_chunk, sample_rate)

                    # Check if we should process
                    if transcriber.should_process():
                        result = transcriber.process_buffer()

                        # Send partial transcription
                        await websocket.send_json({
                            "type": "partial",
                            **result
                        })

                        # Try to polish in background if enabled and text is long enough
                        if enable_polish and len(transcriber.full_transcript) > last_polished_length + 50:
                            try:
                                polished = await transcriber.polish_text_async(transcriber.full_transcript)
                                if polished and polished != transcriber.full_transcript:
                                    transcriber.polished_transcript = polished
                                    await websocket.send_json({
                                        "type": "polished",
                                        "polished_transcript": polished
                                    })
                                    last_polished_length = len(transcriber.full_transcript)
                            except Exception as e:
                                logger.warning(f"Polish error: {e}")

                elif "text" in message:
                    # JSON control message
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON message: {e}")
                        continue

                    if data.get("type") == "end":
                        # Client signaled end of stream
                        logger.info("Client signaled end of stream")
                        break

                    elif data.get("type") == "ping":
                        # Keep-alive ping
                        await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                # Send ping to check connection
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    logger.warning(f"Ping failed, closing connection from {client_ip}")
                    break
            except WebSocketDisconnect:
                logger.info(f"Client disconnected during streaming from {client_ip}")
                connection_closed = True
                break
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                # Continue processing, don't break on single chunk error
                continue

        # Finalize transcription - process any remaining audio even if short
        if transcriber:
            try:
                logger.info(f"Finalizing transcription. Buffer duration: {transcriber.get_buffer_duration():.2f}s, New audio: {transcriber.get_new_audio_duration():.2f}s")
                final_result = transcriber.finalize()
            except Exception as e:
                logger.error(f"Finalization error: {e}")
                final_result = {
                    "full_transcript": transcriber.full_transcript if transcriber else "",
                    "total_duration": "0s",
                    "language": "unknown"
                }

            # Final polish
            if enable_polish and transcriber.full_transcript:
                try:
                    final_polished = await transcriber.polish_text_async(transcriber.full_transcript)
                    if final_polished:
                        transcriber.polished_transcript = final_polished
                        final_result["polished_transcript"] = final_polished
                except Exception as e:
                    logger.warning(f"Final polish error: {e}")

            # Save audio and transcription if we have content
            if transcriber.full_transcript.strip():
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Combine all audio chunks and save
                    if all_audio_chunks:
                        combined_audio = b''.join(all_audio_chunks)
                        audio_filename = f"{timestamp}_streaming.wav"
                        audio_path = Config.RECORDED_AUDIO_DIR / audio_filename

                        # Convert to proper WAV file
                        audio_array = np.frombuffer(combined_audio, dtype=np.int16).astype(np.float32) / 32768.0
                        if sample_rate != 16000:
                            try:
                                from scipy import signal
                                num_samples = int(len(audio_array) * 16000 / sample_rate)
                                audio_array = signal.resample(audio_array, num_samples)
                            except ImportError:
                                logger.warning("scipy not available, saving at original sample rate")
                        sf.write(str(audio_path), audio_array, 16000)

                        # Save transcription
                        save_transcription(
                            transcriber.full_transcript,
                            "streaming.wav",
                            final_result.get("language", "unknown"),
                            final_result.get("total_duration", "0s"),
                            transcriber.segments,
                            timestamp,
                            client_ip
                        )
                except Exception as e:
                    logger.error(f"Error saving audio/transcription: {e}")

            # Send final result
            try:
                await websocket.send_json({
                    "type": "final",
                    **final_result
                })
            except Exception as e:
                logger.error(f"Error sending final result: {e}")

            logger.info(f"WebSocket transcription completed for {client_ip}")
        else:
            logger.warning(f"No transcriber initialized for {client_ip}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from {client_ip}")
    except Exception as e:
        import traceback
        logger.error(f"WebSocket error from {client_ip}: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except Exception:
            logger.error("Failed to send error message to client")
    finally:
        # Cleanup
        try:
            if transcriber:
                del transcriber
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )