# VUI FastAPI Server

A high-performance OpenAI-compatible Text-to-Speech API server powered by [VUI](https://github.com/fluxions-ai/vui) - a small conversational speech model that runs on device.

## Features

- 🎯 **OpenAI-compatible API** - Drop-in replacement for OpenAI's TTS API
- 🚀 **High Performance** - GPU acceleration with optional `torch.compile()` optimization
- 🎵 **Multiple Audio Formats** - WAV, MP3, Opus, FLAC, AAC, PCM support
- 📡 **Streaming Support** - Real-time audio streaming capabilities
- 🐳 **Docker Ready** - Easy deployment with CUDA support
- 💬 **Conversational Quality** - Natural speech with human-like characteristics

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Hugging Face account with accepted terms for:
  - [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)
  - [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/dwain-barnes/vui-fastapi-server.git
cd vui-fastapi-server
```

2. **Get your Hugging Face token**
   - Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a new token with read permissions
   - Accept the terms for the required models (links above)

3. **Build the Docker image**
```bash
docker build --build-arg HUGGING_FACE_HUB_TOKEN=your_hf_token_here -t vui-fastapi .
```

4. **Run the server**
```bash
docker run --gpus all -p 8000:8000 -e USE_GPU=1 vui-fastapi
```

The server will be available at `http://localhost:8000`

## API Usage

### OpenAI-Compatible Endpoint

The server implements the OpenAI Text-to-Speech API specification:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vui",
    "input": "Hello world! This is a test of the VUI text-to-speech system.",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"vui"` | Model identifier (currently only "vui" supported) |
| `input` | string | **required** | Text to convert to speech (max 4096 characters) |
| `voice` | string | `null` | Voice selection (currently ignored) |
| `response_format` | string | `"wav"` | Audio format: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm` |
| `speed` | float | `1.0` | Speech speed (currently ignored) |
| `stream` | boolean | `false` | Enable streaming response |

### Examples

**Basic usage:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "response_format": "mp3"}' \
  --output hello.mp3
```

**Streaming response:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "This is a streaming test", "stream": true}' \
  --output stream.wav
```

**Python example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "vui",
        "input": "Hello from Python!",
        "response_format": "wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Development

### Project Structure

```
vui-fastapi-server/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── download_pyannote.py    # PyAnnote model downloader
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

### Local Development

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Install VUI**
```bash
git clone https://github.com/fluxions-ai/vui.git
pip install -e ./vui
```

3. **Set environment variables**
```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
export USE_GPU=1  # Optional: force GPU usage
```

4. **Run the server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Configuration

Environment variables:
- `USE_GPU`: Set to `1` to force GPU usage, `0` for CPU
- `HUGGING_FACE_HUB_TOKEN`: Required for downloading gated models

## Performance

The server includes several optimizations:

- **GPU Acceleration**: Automatic CUDA detection and usage
- **Model Compilation**: `torch.compile()` for improved inference speed
- **Model Warmup**: Pre-loads and warms up the model during startup
- **Streaming**: Chunked transfer encoding for real-time audio streaming

Typical performance on modern GPUs:
- **Generation Speed**: 1-5x real-time depending on text length
- **Latency**: ~1-3 seconds for short phrases
- **Quality**: High-quality conversational speech with natural characteristics

## API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **OpenAPI spec**: http://localhost:8000/v1/openapi.json

## Troubleshooting

### Common Issues

**Model compilation errors:**
```
RuntimeError: Failed to find C compiler
```
The Dockerfile includes build tools to resolve this. If building locally, install:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential gcc g++

# macOS
xcode-select --install
```

**GPU not detected:**
```
VUI model loaded on cpu
```
Ensure NVIDIA Container Toolkit is installed and `--gpus all` flag is used.

**Hugging Face token errors:**
```
Token is required but no token found
```
Make sure you've accepted the terms for the required models and provided a valid token.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is built on top of [VUI](https://github.com/fluxions-ai/vui) by Fluxions AI. Please refer to the original VUI repository for licensing information.

## Acknowledgments

- [Fluxions AI](https://github.com/fluxions-ai) for the VUI model
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for voice activity detection
- OpenAI for the TTS API specification

## Links

- [VUI Model](https://huggingface.co/fluxions/vui)
- [VUI Demo](https://huggingface.co/spaces/fluxions/vui-space)
- [Original VUI Repository](https://github.com/fluxions-ai/vui)
