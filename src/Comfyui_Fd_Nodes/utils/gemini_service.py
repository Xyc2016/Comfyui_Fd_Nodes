import hashlib
import logging
from io import BytesIO
from typing import Callable, Optional

import numpy as np
import requests
import torch
from PIL import Image

from ..config import (
    FD_GEMINI_URL,
    FD_OSS_URL_PATH_PREFIX_GEMINI,
)
from .error_utils import ERROR_TIMEOUT, normalize_error_message
from .oss_client import upload_bytes_to_oss

logger = logging.getLogger(__name__)


MODEL_NAME_MAP = {
    "gemini-2.5-flash-image-preview": "google/gemini-2.5-flash-image-preview",
    "gemini-3-pro-image-preview": "google/gemini-3-pro-image-preview",
    "batch/gemini-3-pro-image-preview": "batch/gemini-3-pro-image-preview",
    "gemini-3-pro-image-preview-aistudio": "google/gemini-3-pro-image-preview-official",
    "gemini-3-pro-image-preview-official": "google/gemini-3-pro-image-preview-official",
    "gemini-3.1-flash-image-preview": "google/gemini-3.1-flash-image-preview",
}


class GeminiImageServiceError(Exception):
    pass


def summarize_text(text: str, max_chars: int = 120) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}..."


def bytes_calculate_hex_md5(img_bytes: bytes, block_size=64 * 1024) -> str:
    md5 = hashlib.md5()
    for i in range(0, len(img_bytes), block_size):
        md5.update(img_bytes[i:i + block_size])
    return md5.hexdigest()


def bytesio_to_image_tensor(image_bytesio: BytesIO, mode: str = "RGB") -> torch.Tensor:
    image = Image.open(image_bytesio).convert(mode)
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


def normalize_gemini_model_name(model: str) -> str:
    model_name = str(model or "").strip()
    if model_name.startswith(("google/", "batch/")):
        return model_name
    return MODEL_NAME_MAP.get(model_name, f"google/{model_name}" if model_name else model_name)


def compose_prompt(prompt: str, system_prompt: str = "") -> str:
    prompt_text = (prompt or "").strip()
    system_text = (system_prompt or "").strip()
    if system_text and prompt_text:
        return f"{system_text}\n\n{prompt_text}"
    return system_text or prompt_text


class GeminiImageServiceClient:
    def __init__(
        self,
        service_url: Optional[str] = None,
        oss_uploader: Optional[Callable[[str, bytes], str]] = None,
    ):
        self.service_url = service_url or FD_GEMINI_URL
        self.oss_uploader = oss_uploader or upload_bytes_to_oss

    def _tensor_to_png_bytes(self, image_tensor) -> bytes:
        if getattr(image_tensor, "ndim", None) == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        max_pixels = 3072 * 3072
        if img.width * img.height > max_pixels:
            scale = (max_pixels / (img.width * img.height)) ** 0.5
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img = img.resize(size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def upload_image(self, image_tensor) -> str:
        image_bytes = self._tensor_to_png_bytes(image_tensor)
        file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_GEMINI}/{bytes_calculate_hex_md5(image_bytes)}.png"
        image_url = self.oss_uploader(file_oss_path, image_bytes)
        print(f"upload {file_oss_path}")
        return image_url

    def upload_images(self, image_tensors) -> list[str]:
        urls = []
        for idx, image_tensor in enumerate(image_tensors):
            logger.info(
                "Gemini service input image %s resolution: %s",
                idx,
                tuple(image_tensor.squeeze().shape),
            )
            urls.append(self.upload_image(image_tensor))
        return urls

    def build_request_body(
        self,
        *,
        prompt: str,
        model: str,
        image_url_list: list[str],
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        out_request_id: str = "",
        enable_color_bias_correction: bool = False,
        color_bias_reference_image_index: int = 0,
    ) -> dict:
        body = {
            "out_request_id": out_request_id or "default",
            "prompt": prompt,
            "model": normalize_gemini_model_name(model),
            "aspect_ratio": aspect_ratio or "",
            "image_url_list": image_url_list,
        }
        if image_size:
            body["resolution"] = image_size
        if enable_color_bias_correction is True:
            body["enable_color_bias_correction"] = True
            body["color_bias_reference_image_index"] = (
                color_bias_reference_image_index
                if isinstance(color_bias_reference_image_index, int)
                and not isinstance(color_bias_reference_image_index, bool)
                else 0
            )
        return body

    def summarize_request_body(self, body: dict) -> dict:
        return {
            "url": self.service_url,
            "out_request_id": body.get("out_request_id"),
            "model": body.get("model"),
            "aspect_ratio": body.get("aspect_ratio"),
            "resolution": body.get("resolution"),
            "prompt_preview": summarize_text(body.get("prompt")),
            "prompt_length": len(body.get("prompt") or ""),
            "image_count": len(body.get("image_url_list") or []),
        }

    def call_with_image_urls(
        self,
        *,
        prompt: str,
        model: str,
        image_url_list: list[str],
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        out_request_id: str = "",
        enable_color_bias_correction: bool = False,
        color_bias_reference_image_index: int = 0,
    ):
        if not self.service_url:
            raise RuntimeError("未配置 Gemini 服务地址，请设置环境变量 FD_GEMINI_URL")
        if not prompt or not prompt.strip():
            raise RuntimeError("prompt 不能为空")
        if not image_url_list:
            raise RuntimeError("未提供图片")

        body = self.build_request_body(
            prompt=prompt.strip(),
            model=model,
            image_url_list=image_url_list,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            out_request_id=out_request_id,
            enable_color_bias_correction=enable_color_bias_correction,
            color_bias_reference_image_index=color_bias_reference_image_index,
        )
        logger.info("Calling Gemini image service with payload=%s", self.summarize_request_body(body))

        response = None
        try:
            response = requests.post(self.service_url, json=body, timeout=600)
            if not response.ok:
                raise RuntimeError(f"API 请求失败: {response.status_code}\n{response.text[:1000]}")
            result = response.json()
            logger.info(
                "Gemini image service response summary: %s",
                {
                    "status_code": response.status_code,
                    "result_image_url": result.get("result_image_url"),
                    "cost_time": result.get("cost_time"),
                    "has_error": bool(result.get("error")),
                },
            )

            if result.get("error", {}).get("code"):
                error = result["error"]
                raise GeminiImageServiceError(
                    normalize_error_message(
                        error.get("message") or error["code"],
                        category=error["code"],
                    )
                )

            result_url = result["result_image_url"]
            image_response = requests.get(result_url, timeout=300)
            image_response.raise_for_status()
            return bytesio_to_image_tensor(BytesIO(image_response.content), mode="RGB"), result_url, result.get("message", "")
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="request timed out")
            ) from exc
        except (KeyError, ValueError) as exc:
            response_text = response.text[:500] if response is not None else ""
            raise RuntimeError(normalize_error_message(f"解析响应失败: {exc}\n原始响应: {response_text}")) from exc
        except GeminiImageServiceError:
            raise
        except Exception as exc:
            raise RuntimeError(normalize_error_message(exc)) from exc

    def call(
        self,
        *,
        prompt: str,
        model: str,
        image_tensors,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        out_request_id: str = "",
        enable_color_bias_correction: bool = False,
        color_bias_reference_image_index: int = 0,
    ):
        image_url_list = self.upload_images(image_tensors)
        return self.call_with_image_urls(
            prompt=prompt,
            model=model,
            image_url_list=image_url_list,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            out_request_id=out_request_id,
            enable_color_bias_correction=enable_color_bias_correction,
            color_bias_reference_image_index=color_bias_reference_image_index,
        )
