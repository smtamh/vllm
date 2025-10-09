import os

# Directories
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, "..", "img")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")

# Prompt
ROLE_PATH  = os.path.join(BASE_DIR, "..", "role.txt")
TOOL_PATH  = os.path.join(BASE_DIR, "..", "tools.json")

# Models
YOLO_PATH  = os.path.join(MODEL_DIR, "yolo_best.pt")
STT_PATH   = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")
DETECTOR   = None
GDINO_PATH = os.path.join(MODEL_DIR, "grounding-dino-base")
SAM2_TYPE  = "sam2.1_hiera_tiny"
SAM2_PATH  = os.path.join(MODEL_DIR, "sam2.1_hiera_tiny.pt")

# Image for tool
CURRENT_IMAGE = None

# LLM & VLM setting
VLM_MODEL_PATH  = os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct-AWQ")
VLM_GPU_UTIL    = 0.8
VLM_MAX_LENGTH  = 16384
VLM_LIMIT_INPUT = {"video": 0}

SAMPLING_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 20,
    "max_tokens": 300
}

# Network Settings
SERVER_ROS_MASTER_URI = "http://10.149.193.0:11311"
SERVER_ROS_IP = "10.149.193.0"
CLIENT_ROS_IP = "10.150.196.89"

# Input Mode
USE_TYPING = True  # If True, use keyboard input; if False, use STT input