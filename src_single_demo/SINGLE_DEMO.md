## Installation
### System Packages
    sudo apt install libportaudio2 libportaudiocpp0 portaudio19-dev ffmpeg espeak-ng

### Python Packages
Install packages in `python 3.10` virtual environment in `Ubuntu 22.04.# LTS`:
    
    uv pip install sounddevice vosk vllm ultralytics
    
#### FlashInfer (Optional)
    
    git submodule update --init --recursive
    cd third_party/flashinfer
    uv pip install -e . --no-build-isolation -v
    
#### Flash-Attn (Optional)
⚠️ DO NOT USE IN UBUNTU 20.04
    
    uv pip install flash-attn --no-build-isolation -v
    

<br>

## Inference
    python chat.py