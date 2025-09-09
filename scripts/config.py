import os

# Directories
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, "..", "img")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
AUDIO_DIR  = os.path.join("/home/smtamh/dyros/chatgpt/speech")

# Prompt
ROLE_PATH  = os.path.join(BASE_DIR, "..", "role.txt")
TOOL_PATH  = os.path.join(BASE_DIR, "..", "tools.json")

# Paths
IMAGE_PATH = None
AUDIO_PATH = None

# Models
YOLO_PATH  = os.path.join(MODEL_DIR, "yolo_best.pt")
STT_PATH   = os.path.join(MODEL_DIR, "vosk-model-small-ko-0.22")

# LLM & VLM setting
VLM_MODEL_PATH  = os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct-AWQ")
# VLM_MODEL_PATH  = "/home/smtamh/models/Qwen2.5-VL-7B-Instruct-AWQ"
VLM_GPU_UTIL    = 0.8
VLM_MAX_LENGTH  = 16384
VLM_LIMIT_INPUT = {"video": 0}

SAMPLING_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 20,
    "max_tokens": 300
}