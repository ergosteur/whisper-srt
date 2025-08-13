# whisper-srt

`whisper-srt` is a command-line tool for automatic speech recognition (ASR) and subtitle generation using OpenAI Whisper models. It transcribes audio/video files and outputs SRT subtitle files.

## Features

- Supports local and HuggingFace Whisper models
- Optional custom weights via safetensors
- Language detection or manual override
- Transcription and translation tasks
- Batch and chunked processing for long audio

## Usage

```
python whisper-srt.py --audio input.mp3 --model-dir ./anime_whisper
python whisper-srt.py --audio input.mp4 --model-id openai/whisper-small --weights anime_whisper.safetensors
```

### Common Arguments

- `--audio`: Path to input audio/video file (required)
- `--model-dir`: Local model directory (preferred)
- `--model-id`: HuggingFace model ID
- `--weights`: Path to custom safetensors weights
- `--language`: Language code (e.g., en, fr, ja; default: auto-detect)
- `--task`: `transcribe` (default) or `translate`
- `--output`: Output SRT file path (default: same as input, with `.srt` extension)

## Environment Setup

Windows (PowerShell):

```pwsh
.\setup_venv.ps1
```

Linux/macOS (Bash):

```bash
./setup_venv.sh
```

## Dependencies

- Python 3.8+
- PyTorch (CPU or CUDA)
- transformers, accelerate, librosa, soundfile, ffmpeg-python, safetensors, tqdm

Install dependencies:

```sh
pip install -r requirements.txt
```
