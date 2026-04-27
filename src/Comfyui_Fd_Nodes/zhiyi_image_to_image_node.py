import requests
import json
import base64
import random
import numpy as np
from PIL import Image
import io
import torch
from .config_manager import load_config


class ZhiYiImageToImageNode:
    """知衣图生图节点 - 最多6张输入图片，batch_size 控制重复请求次数"""

    MODELS = [
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
    ]

    ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4"]
    IMAGE_SIZES = ["4K", "2K", "1080P", "720P"]
    SEED_MODES = ["随机种子", "固定种子"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "将裤子变为老欧美风格的牛仔裤",
                    "multiline": True,
                }),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "1:1"}),
                "image_size": (cls.IMAGE_SIZES, {"default": "4K"}),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "seed_mode": (cls.SEED_MODES, {"default": "随机种子"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                }),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
    FUNCTION = "generate"
    CATEGORY = "知衣/图生图"
    OUTPUT_NODE = False

    def _tensor_to_base64(self, image_tensor):
        arr = (image_tensor[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _base64_to_tensor(self, data_url):
        if ";base64," in data_url:
            _, b64 = data_url.split(";base64,", 1)
        else:
            b64 = data_url
        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _extract_image_from_response(self, result):
        choice = result["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            raise RuntimeError("内容被过滤 (content_filter)，请修改提示词或输入图片后重试")
        msg = choice["message"]

        # 格式1: message.images[0].image_url.url
        if "images" in msg and msg["images"]:
            return msg["images"][0]["image_url"]["url"]

        # 格式2: message.content 列表中含 image_url 类型
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        return part["image_url"]["url"]
                    # 格式3: type=image, 直接含 base64
                    if part.get("type") == "image":
                        return part.get("url") or part.get("data", "")

        # 格式4: content 直接是 base64 字符串
        if isinstance(content, str) and len(content) > 100:
            return content

        raise KeyError(f"无法从响应中提取图片，响应结构: {list(msg.keys())}")

    # gemini image-preview 系列不支持 seed 参数
    MODELS_NO_SEED = {"gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"}

    def _single_request(self, url, api_key, messages, model, aspect_ratio, image_size, seed):
        payload = {
            "stream": False,
            "model": model,
            "messages": messages,
            "imageConfig": {"aspect_ratio": aspect_ratio, "image_size": image_size},
            "modalities": ["image"],
        }
        if model not in self.MODELS_NO_SEED:
            payload["seed"] = seed
        response = requests.post(
            url=url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=600,
        )
        if not response.ok:
            raise RuntimeError(f"API 请求失败: {response.status_code}\n{response.text[:1000]}")
        result = response.json()
        try:
            out_data_url = self._extract_image_from_response(result)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"解析响应失败: {e}\n原始响应: {response.text[:500]}")
        return self._base64_to_tensor(out_data_url)

    def generate(self, image_1, prompt, model, aspect_ratio, image_size,
                 batch_size=1, seed_mode="随机种子", seed=0,
                 image_2=None, image_3=None, image_4=None,
                 image_5=None, image_6=None, system_prompt=""):
        cfg = load_config()
        base_url = cfg["base_url"]
        api_key = cfg["api_key"]
        base_url = base_url.rstrip("/")
        url = f"{base_url}/v1/chat/completions"

        actual_seed = random.randint(0, 2147483647) if seed_mode == "随机种子" else seed

        images = [t for t in [image_1, image_2, image_3, image_4, image_5, image_6] if t is not None]

        messages = []
        if system_prompt.strip():
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        user_content = [{"type": "text", "text": prompt}]
        for img_tensor in images:
            b64 = self._tensor_to_base64(img_tensor)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        messages.append({"role": "user", "content": user_content})

        results = []
        for i in range(batch_size):
            batch_seed = actual_seed + i if seed_mode == "固定种子" else random.randint(0, 2147483647)
            tensor = self._single_request(url, api_key, messages, model, aspect_ratio, image_size, batch_seed)
            results.append(tensor)

        return (torch.cat(results, dim=0), actual_seed)
