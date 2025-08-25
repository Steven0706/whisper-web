#!/usr/bin/env python3
"""
Whisper Web - Self-hosted Speech Recognition Service
High-performance voice-to-text transcription using OpenAI's Whisper
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
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

def save_transcription(text: str, filename: str, language: str, processing_time: str, segments: list = None):
    """Save transcription to history"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    transcription_data = {
        "timestamp": timestamp,
        "filename": filename,
        "text": text[:500],  # Store preview
        "full_text": text,
        "language": language,
        "processing_time": processing_time,
        "model": MODEL_NAME,
        "segments": segments[:10] if segments else [],  # Store first 10 segments
        "audio_file": f"{timestamp}_{filename}"
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
        
        # Save transcription
        save_transcription(
            result["text"],
            actual_filename,
            result.get("language", "unknown"),
            processing_time,
            result.get("segments", [])
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
    limit: int = 20,
    token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get recent transcription history (requires authentication)"""
    
    # Check authentication
    if not token or not verify_session(token):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        history = []
        files = sorted(Config.TRANSCRIPTION_DIR.glob("*.json"), reverse=True)[:limit]
        
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                history.append(json.load(f))
        
        return history
    except Exception as e:
        logger.error(f"Error loading history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load history")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )