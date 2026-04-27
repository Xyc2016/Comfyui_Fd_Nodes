import requests
import base64
import io
import math
import numpy as np
import torch
from PIL import Image
from .config_manager import load_config

# 长边像素数对应的档位
_RESOLUTION = {"1K": 1024, "2K": 2048, "4K": 4096}

# 比例 -> (w_ratio, h_ratio)
_ASPECT = {
    "1:1":  (1, 1),
    "3:2":  (3, 2), "2:3":  (2, 3),
    "16:9": (16, 9), "9:16": (9, 16),
    "5:4":  (5, 4), "4:5":  (4, 5),
    "4:3":  (4, 3), "3:4":  (3, 4),
    "21:9": (21, 9), "9:21": (9, 21),
    "1:3":  (1, 3), "3:1":  (3, 1),
    "2:1":  (2, 1), "1:2":  (1, 2),
    "1:8":  (1, 8), "8:1":  (8, 1),
}

def _calc_size(aspect: str, resolution: str) -> str:
    w_r, h_r = _ASPECT[aspect]
    long = _RESOLUTION[resolution]
    if w_r >= h_r:
        w = long
        h = round(long * h_r / w_r / 8) * 8
    else:
        h = long
        w = round(long * w_r / h_r / 8) * 8
    return f"{w}x{h}"


class GPTImageEditNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit_image"
    CATEGORY = "GPT Image"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "生成狗狗趴在草地上的近景画面",
                    "multiline": True,
                    "tooltip": "Editing prompt"
                }),
                "aspect_ratio": (list(_ASPECT.keys()), {
                    "default": "1:1",
                    "tooltip": "输出图像比例"
                }),
                "resolution": (list(_RESOLUTION.keys()), {
                    "default": "1K",
                    "tooltip": "长边像素档位：1K=1024, 2K=2048, 4K=4096"
                }),
                "quality": (["auto", "low", "medium", "high"], {
                    "default": "auto"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of images to generate"
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 30,
                    "max": 1200,
                    "step": 30,
                    "tooltip": "Read timeout in seconds"
                }),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "Input image 1"}),
                "image2": ("IMAGE", {"tooltip": "Input image 2"}),
                "image3": ("IMAGE", {"tooltip": "Input image 3"}),
                "image4": ("IMAGE", {"tooltip": "Input image 4"}),
            }
        }

    def _session(self):
        session = requests.Session()
        session.trust_env = True  # 读取 HTTPS_PROXY / HTTP_PROXY 环境变量
        return session

    def _tensor_to_png(self, tensor):
        img_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _decode_response(self, data):
        img_data = data["data"][0]
        if "b64_json" in img_data:
            return base64.b64decode(img_data["b64_json"])
        elif "url" in img_data:
            return requests.get(img_data["url"], timeout=60).content
        raise ValueError(f"Unexpected response format: {list(img_data.keys())}")

    def edit_image(self, prompt, aspect_ratio, resolution, quality,
                   batch_size=1, timeout=300,
                   image1=None, image2=None, image3=None, image4=None):
        cfg = load_config()
        base_url = cfg["base_url"]
        api_key = cfg["api_key"]
        size = _calc_size(aspect_ratio, resolution)
        url = base_url.rstrip("/") + "/v1/images/edits"
        headers = {"Authorization": f"Bearer {api_key}"}
        session = self._session()

        input_images = [img for img in [image1, image2, image3, image4] if img is not None]

        results = []
        for _ in range(batch_size):
            files = [
                ("model", (None, "gpt-image-2")),
                ("prompt", (None, prompt)),
                ("size", (None, size)),
                ("quality", (None, quality)),
            ]

            for img_tensor in input_images:
                buf = self._tensor_to_png(img_tensor)
                files.append(("image", ("image.png", buf, "image/png")))

            response = session.post(url, headers=headers, files=files,
                                    timeout=(30, timeout))
            response.raise_for_status()

            img_bytes = self._decode_response(response.json())
            pil_out = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            arr = np.array(pil_out).astype(np.float32) / 255.0
            results.append(torch.from_numpy(arr))

        return (torch.stack(results),)
