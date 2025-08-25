# 🎙️ Whisper Web - Self-Hosted Speech Recognition Service

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange)](https://github.com/openai/whisper)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

A high-performance, self-hosted speech-to-text service powered by OpenAI's Whisper model. Features a modern web interface, REST API, and GPU acceleration support.

[English](#english) | [中文](#中文)

![Demo](docs/demo.gif)

<img width="1535" height="1260" alt="image" src="https://github.com/user-attachments/assets/8386fc89-1873-451a-914c-c0bba73dabeb" />


## ✨ Features

- 🚀 **GPU Acceleration** - CUDA support for 10x faster transcription
- 🌐 **Web Interface** - Modern, responsive UI with drag-and-drop
- 🎤 **Browser Recording** - Direct microphone recording (hold spacebar)
- 📡 **REST API** - Full-featured API for integration
- 🔒 **Privacy First** - All processing done locally, no cloud dependencies
- 🌍 **Multi-language** - Support for 100+ languages with auto-detection
- 📝 **History Tracking** - Optional password-protected transcription history
- 🔄 **Auto-start** - Systemd service for production deployment
- 🎯 **CORS Support** - Embed in other web applications

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (optional, for acceleration)
- 4GB+ RAM (8GB+ recommended)
- Ubuntu 20.04+ or similar Linux distribution

### One-Line Installation

```bash
curl -sSL https://raw.githubusercontent.com/Steven0706/whisper-web/main/install.sh | bash
```

### Manual Installation

1. **Clone the repository**
```bash
git clone https://github.com/Steven0706/whisper-web.git
cd whisper-web
```

2. **Run setup script**
```bash
chmod +x setup.sh
./setup.sh
```

3. **Configure environment**
```bash
cp .env.example .env
nano .env  # Edit configuration
```

4. **Start the service**
```bash
# Development
python app.py

# Production (with systemd)
sudo systemctl start whisper-web
sudo systemctl enable whisper-web
```

5. **Access the service**
- Web Interface: http://localhost:5000
- API Documentation: http://localhost:5000/docs

## 🎯 Usage Examples

### Web Interface
1. Open http://localhost:5000 in your browser
2. Click and hold spacebar to record, or drag and drop audio files
3. View real-time transcription results

### API Usage

**Transcribe Audio File**
```bash
curl -X POST http://localhost:5000/api/transcribe \
  -F "audio=@speech.wav" \
  -F "language=en"
```

**Transcribe Base64 Audio**
```javascript
const response = await fetch('http://localhost:5000/api/transcribe/json', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        audio: base64AudioData,
        language: 'auto'
    })
});
```

**Python Client**
```python
import requests

def transcribe(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/transcribe',
            files={'audio': f}
        )
    return response.json()['text']
```

## ⚙️ Configuration

Edit `.env` file to customize:

```env
# Server
HOST=0.0.0.0
PORT=5000

# Model (tiny, base, small, medium, large, large-v3)
WHISPER_MODEL=base

# Authentication
AUTH_PASSWORD=your_secure_password

# Performance
DEVICE=auto  # auto, cuda, cpu
```

### Available Models

| Model | VRAM | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | ~1GB | ⚡⚡⚡⚡⚡ | ⭐⭐ | Real-time, low accuracy |
| base | ~1GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Fast transcription |
| small | ~2GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| medium | ~5GB | ⚡⚡ | ⭐⭐⭐⭐ | Good accuracy |
| large-v3 | ~10GB | ⚡ | ⭐⭐⭐⭐⭐ | Best accuracy |

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/status` | Server status and GPU info |
| POST | `/api/transcribe` | Transcribe audio (multipart) |
| POST | `/api/transcribe/json` | Transcribe audio (JSON/base64) |
| POST | `/api/auth` | Authenticate for history |
| GET | `/api/history` | Get transcription history |

### Response Format
```json
{
  "success": true,
  "text": "Transcribed text here",
  "language": "en",
  "processing_time": "1.23s",
  "segments": [...],
  "device": "cuda",
  "model": "base"
}
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t whisper-web .

# Run container
docker run -d \
  --gpus all \
  -p 5000:5000 \
  -v whisper-data:/app/data \
  --name whisper \
  whisper-web
```

## 🔧 Advanced Setup

### NVIDIA GPU Setup
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

# Verify installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name whisper.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### SSL/HTTPS Setup
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d whisper.yourdomain.com
```

## 📊 Performance Benchmarks

| Hardware | Model | Audio Length | Processing Time |
|----------|-------|--------------|-----------------|
| RTX 4090 | large-v3 | 60s | ~1.5s |
| RTX 3060 | base | 60s | ~1.0s |
| CPU (i7-12700) | base | 60s | ~15s |

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
```bash
# Use smaller model
export WHISPER_MODEL=base
# Or force CPU mode
export DEVICE=cpu
```

### Port Already in Use
```bash
# Change port in .env
PORT=5001
# Or kill existing process
sudo lsof -i :5000
sudo kill -9 <PID>
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - The amazing speech recognition model
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/Steven0706/whisper-web/issues)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Steven0706/whisper-web&type=Date)](https://star-history.com/#Steven0706/whisper-web&Date)

---

<div align="center">
Made with ❤️ by the community
</div>

---

# 中文

## 🎙️ Whisper Web - 自托管语音识别服务

一个高性能的自托管语音转文字服务，基于 OpenAI 的 Whisper 模型。具有现代化的 Web 界面、REST API 和 GPU 加速支持。

## ✨ 特性

- 🚀 **GPU 加速** - CUDA 支持，速度提升 10 倍
- 🌐 **Web 界面** - 现代化响应式 UI，支持拖放
- 🎤 **浏览器录音** - 直接麦克风录音（按住空格键）
- 📡 **REST API** - 功能完整的 API 接口
- 🔒 **隐私优先** - 所有处理都在本地完成，无云端依赖
- 🌍 **多语言支持** - 支持 100+ 种语言，自动检测
- 📝 **历史记录** - 可选的密码保护转写历史
- 🔄 **自动启动** - Systemd 服务用于生产部署
- 🎯 **CORS 支持** - 可嵌入其他 Web 应用

## 🚀 快速开始

详细的中文安装和使用说明请参考上方英文文档，配置过程完全相同。

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
