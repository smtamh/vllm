# Environment Setting
- Python 3.10

## 1. vlm
- uv pip install vllm

## 2. sound / audio
- uv pip install sounddevice    # mic
- uv pip install pydub          # .mp3 -> .wav
- uv pip install vosk           # stt model

## 3. vision
- uv pip install ultralytics    # YOLO

## 4. flashinfer
- cd third_party/flashinfer                         # git submodule
- uv pip install -e . --no-build-isolation -v

## 5. flash-attn
- uv pip install flash-attn --no-build-isolation -v

# Execution
- python chat.py
