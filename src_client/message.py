# UTILS
import json
from collections import deque

class MsgBuffer:
    def __init__(self, system_prompt, max_groups=5):
        self.system = {"role": "system", "content": system_prompt}
        self.groups = deque([], maxlen=max_groups)

    def start_user(self, text, image_pil=None):
        # add new user input
        # add image if exists
        # strip previous images

        content = []
        if image_pil is not None:
            content.append({"type": "image_pil", "image_pil": image_pil})
        content.append({"type": "text", "text": text})
        self.groups.append([{"role": "user", "content": content}])
        self._strip_images()

    def append_assistant(self, text):
        # add assistant responses

        self.groups[-1].append({"role": "assistant", "content": text})

    def append_tool(self, results):
        # add tool responses
        
        text = "\n".join(json.dumps(d, ensure_ascii=False) for d in results)
        self.groups[-1].append({"role": "tool", "content": f"<tool_response>\n{text}\n</tool_response>"})

    def update_image(self, image_pil):
        # for assistant responses after tool calling

        first = self.groups[-1][0]
        if isinstance(first["content"], list):
            first["content"] = [c for c in first["content"] if c.get("type") != "image_pil"]
            first["content"].insert(0, {"type": "image_pil", "image_pil": image_pil})
        else:
            first["content"] = [
                {"type": "image_pil", "image_pil": image_pil},
                {"type": "text", "text": first["content"]}
            ]

    def _strip_images(self):
        # delete previous image inputs
        for idx, group in enumerate(self.groups):
            if idx == len(self.groups) - 1:
                continue
            if group and group[0]["role"] == "user" and isinstance(group[0]["content"], list):
                group[0]["content"] = [c for c in group[0]["content"] if c.get("type") != "image_pil"]

    def to_messages(self):
        # convert to list of messages for vllm.chat()
        messages = [self.system]
        for group in self.groups:
            messages.extend(group)
        return messages

    def __str__(self):
        result = []
        result.append("System:")
        result.append(self.system['content'])
        result.append("=" * 50)
        
        for i, group in enumerate(self.groups):
            result.append(f"Group {i+1}:")
            for msg in group:
                role = msg['role']
                content = msg['content']
                
                if isinstance(content, list):
                    # image + text
                    text_parts = [c['text'] for c in content if c.get('type') == 'text']
                    image_parts = [c for c in content if c.get('type') == 'image_pil']
                    
                    content_str = '\n'.join(text_parts)
                    if image_parts:
                        content_str += f" [contains {len(image_parts)} image(s)]"
                else:
                    content_str = content
                
                result.append(f"{role}:")
                result.append(content_str)
                result.append("")
            result.append("-" * 30)
        
        return '\n'.join(result)