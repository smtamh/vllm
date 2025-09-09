import json, os, cv2

import config

def build_prompt(role=config.ROLE_PATH, tool=config.TOOL_PATH):

    # system prompt: .txt
    with open(role, "r", encoding="utf-8") as f:
        role_text = f.read()

    # tools: .json
    with open(tool, "r", encoding="utf-8") as f:
        tool_data = json.load(f)

    # convert to string
    tool_items = [json.dumps(item, ensure_ascii=False, separators=(",",":")) for item in tool_data]
    tools_str = "\n".join(tool_items)

    system_prompt = role_text.replace("{{TOOLS_JSON}}", tools_str)
    return system_prompt


def center_crop(img, size=320):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized = cv2.resize(cropped, (size, size))
    return resized