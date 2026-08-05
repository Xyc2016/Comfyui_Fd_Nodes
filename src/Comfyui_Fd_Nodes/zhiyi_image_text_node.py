import requests
import json
import base64
import logging
import numpy as np
from PIL import Image
import io
from .config_manager import load_config
from .utils.logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)

MAX_IMAGE_DATA_URL_BYTES = 10_000_000
MAX_IMAGE_TOTAL_PIXELS = 35_000_000
JPEG_QUALITY_STEPS = (85, 80, 75, 70, 65, 60)
MIN_IMAGE_LONG_EDGE = 1024
IMAGE_RESIZE_FACTOR = 0.8


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
                "node_switch": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "number",
                }),
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

    def _image_tensor_to_pil(self, image_tensor):
        if getattr(image_tensor, "ndim", None) == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def _encode_jpeg_data_url(self, pil_img, quality):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_b64}", len(buf.getvalue())

    def _resize_to_long_edge(self, pil_img, long_edge):
        width, height = pil_img.size
        current_long_edge = max(width, height)
        if current_long_edge <= long_edge:
            return pil_img

        scale = long_edge / current_long_edge
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)

    def _resize_to_max_pixels(self, pil_img, max_pixels):
        width, height = pil_img.size
        total_pixels = width * height
        if total_pixels <= max_pixels:
            return pil_img

        scale = (max_pixels / total_pixels) ** 0.5
        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)

    def _image_tensor_to_data_url(self, image_tensor):
        original_img = self._image_tensor_to_pil(image_tensor)
        original_size = original_img.size
        original_pixels = original_size[0] * original_size[1]
        source_img = self._resize_to_max_pixels(original_img, MAX_IMAGE_TOTAL_PIXELS)
        source_size = source_img.size
        resized_by_pixels = source_size != original_size
        current_long_edge = max(source_size)
        best_result = None

        while True:
            candidate_img = self._resize_to_long_edge(source_img, current_long_edge)
            final_pixels = candidate_img.size[0] * candidate_img.size[1]
            for quality in JPEG_QUALITY_STEPS:
                data_url, image_bytes = self._encode_jpeg_data_url(candidate_img, quality)
                data_url_bytes = len(data_url.encode("utf-8"))
                info = {
                    "original_size": original_size,
                    "final_size": candidate_img.size,
                    "original_pixels": original_pixels,
                    "final_pixels": final_pixels,
                    "max_total_pixels": MAX_IMAGE_TOTAL_PIXELS,
                    "resized_by_pixels": resized_by_pixels,
                    "mime_type": "image/jpeg",
                    "quality": quality,
                    "image_bytes": image_bytes,
                    "data_url_bytes": data_url_bytes,
                    "max_data_url_bytes": MAX_IMAGE_DATA_URL_BYTES,
                    "resized": candidate_img.size != original_size,
                }
                best_result = (data_url, info)
                if data_url_bytes <= MAX_IMAGE_DATA_URL_BYTES:
                    return best_result

            if current_long_edge <= MIN_IMAGE_LONG_EDGE:
                break
            current_long_edge = max(MIN_IMAGE_LONG_EDGE, int(current_long_edge * IMAGE_RESIZE_FACTOR))

        _, info = best_result
        raise RuntimeError(
            "图生文输入图片压缩后仍超过 10MB 传输限制: "
            f"{info['data_url_bytes']} bytes > {MAX_IMAGE_DATA_URL_BYTES} bytes; "
            "请降低输入图片分辨率后重试"
        )

    def _summarize_messages_for_log(self, messages):
        summarized_messages = []
        for message in messages:
            summarized_message = {"role": message.get("role", "")}
            content = message.get("content", [])
            if isinstance(content, list):
                text_parts = []
                image_count = 0
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text = part.get("text", "").strip()
                        if text:
                            text_parts.append(text)
                    elif part.get("type") in {"image_url", "image"}:
                        image_count += 1
                if text_parts:
                    summarized_message["text"] = "\n".join(text_parts)
                if image_count:
                    summarized_message["image_count"] = image_count
            else:
                summarized_message["content"] = content
            summarized_messages.append(summarized_message)
        return summarized_messages

    def _summarize_response_for_log(self, result):
        summary = {"keys": list(result.keys())}
        choices = result.get("choices", [])
        summary["choice_count"] = len(choices)
        if not choices:
            return summary

        message = choices[0].get("message", {})
        summary["message_keys"] = list(message.keys())
        content = message.get("content", [])
        if isinstance(content, list):
            summary["text_part_count"] = sum(
                1 for part in content if isinstance(part, dict) and part.get("type") == "text"
            )
        elif isinstance(content, str):
            summary["content_length"] = len(content)
        return summary

    def generate(self, image, prompt, node_switch=1,
                 system_prompt="", temperature=0.7, max_tokens=2048):
        if node_switch == 1:
            return ("",)

        cfg = load_config()
        base_url = cfg["base_url"]
        api_key = cfg["api_key"]
        base_url = base_url.rstrip("/")
        url = f"{base_url}/v1/chat/completions"

        data_url, image_info = self._image_tensor_to_data_url(image)
        logger.info("ZhiYi image-to-text encoded image: %s", image_info)

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
            "model": "doubao-seed-2.0-mini",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.info(
            "Calling ZhiYi image-to-text API with payload=%s",
            {
                "url": url,
                "stream": payload["stream"],
                "model": payload["model"],
                "temperature": payload["temperature"],
                "max_tokens": payload["max_tokens"],
                "messages": self._summarize_messages_for_log(messages),
            },
        )

        try:
            response = requests.post(
                url=url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
                data=json.dumps(payload),
                timeout=600,
            )
            if not response.ok:
                raise RuntimeError(f"API 请求失败: {response.status_code}\n{response.text[:1000]}")
            result = response.json()
            logger.info(
                "ZhiYi image-to-text API response summary: %s",
                {
                    "status_code": response.status_code,
                    **self._summarize_response_for_log(result),
                },
            )
            text = result["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )
            return (text,)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API 请求失败: {e}")
        except RuntimeError:
            raise
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"解析响应失败: {e}\n原始响应: {response.text[:500]}")
