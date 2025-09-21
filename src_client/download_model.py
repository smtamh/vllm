# After download the model, delete in hugging face hub: ~/.cache/huggingface/hub

# Download from huggingface
from huggingface_hub import snapshot_download
import config, os

model_id = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

local_dir = os.path.join(config.MODEL_DIR, model_id.split('/')[-1])

model_path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
)