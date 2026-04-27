import requests
import json
import base64
import numpy as np
from PIL import Image
import io
from .config_manager import load_config


class ZhiYiImageTextNode:
    """知衣图生文节点 - 传入图片，调用 Gemini 模型生成描述文本"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "介绍这幅图片",
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
    CATEGORY = "知衣/图生文"
    OUTPUT_NODE = False

    def _image_tensor_to_base64(self, image_tensor):
        # ComfyUI IMAGE: shape (B, H, W, C), float32, 0~1
        arr = (image_tensor[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(self, image, prompt,
                 system_prompt="", temperature=0.7, max_tokens=2048):
        cfg = load_config()
        base_url = cfg["base_url"]
        api_key = cfg["api_key"]
        base_url = base_url.rstrip("/")
        url = f"{base_url}/v1/chat/completions"

        img_b64 = self._image_tensor_to_base64(image)
        data_url = f"data:image/png;base64,{img_b64}"

        messages = []
        if system_prompt.strip():
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
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
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )
            return (text,)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API 请求失败: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"解析响应失败: {e}\n原始响应: {response.text[:500]}")
