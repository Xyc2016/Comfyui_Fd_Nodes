import requests
import json
from .config_manager import load_config


class ZhiYiTextGenNode:
    """知依文生文节点 - 调用 Gemini 模型生成文本"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "请介绍你自己",
                    "multiline": True,
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "知衣/文生文"
    OUTPUT_NODE = False

    def generate(self, prompt, system_prompt="", temperature=0.7, max_tokens=2048):
        cfg = load_config()
        base_url = cfg["base_url"]
        api_key = cfg["api_key"]
        base_url = base_url.rstrip("/")
        url = f"{base_url}/v1/chat/completions"

        messages = []
        if system_prompt.strip():
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })

        payload = {
            "stream": False,
            "model": "gemini-3-pro-preview",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                url=url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=600,
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            # content 可能是字符串或列表
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )
            return (text,)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API 请求失败: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"解析响应失败: {e}\n原始响应: {response.text[:500]}")
