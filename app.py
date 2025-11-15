#!/usr/bin/env python3
"""
Whisper Web - Self-hosted Speech Recognition Service
High-performance voice-to-text transcription using OpenAI's Whisper
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
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

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )