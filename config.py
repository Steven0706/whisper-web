"""Configuration management for Whisper server"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    
    # Model settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    DEVICE = os.getenv("DEVICE", "auto")
    
    # Authentication
    AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "change_me_please")
    SESSION_DURATION_HOURS = int(os.getenv("SESSION_DURATION_HOURS", 24))
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 30))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
    
    # File settings
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 100))
    MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    TRANSCRIPTION_DIR = BASE_DIR / "transcriptions"
    RECORDED_AUDIO_DIR = BASE_DIR / "recorded_audio"
    TEMPLATE_DIR = BASE_DIR / "templates"
    CLIPBOARD_DIR = BASE_DIR / "clipboard"
    
    # Allowed audio types
    ALLOWED_AUDIO_TYPES = {
        'audio/wav', 'audio/x-wav', 'audio/wave',
        'audio/mp3', 'audio/mpeg',
        'audio/mp4', 'audio/m4a', 'audio/x-m4a',
        'audio/ogg', 'audio/webm',
        'audio/flac', 'audio/x-flac',
        'video/mp4', 'video/webm'
    }
    
    @classmethod
    def init_dirs(cls):
        """Create necessary directories"""
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.TRANSCRIPTION_DIR.mkdir(exist_ok=True)
        cls.RECORDED_AUDIO_DIR.mkdir(exist_ok=True)
        cls.TEMPLATE_DIR.mkdir(exist_ok=True)
        cls.CLIPBOARD_DIR.mkdir(exist_ok=True)