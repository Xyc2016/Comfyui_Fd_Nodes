import io
import logging
import traceback
from io import BytesIO
from typing import List, Optional, Tuple

import requests
from PIL import Image

from ..config import (
    FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL,
    FD_GPT_IMAGE_EDIT_URL,
    FD_IMAGE_GENERATE_URL,
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
    FD_OSS_URL_PATH_PREFIX_BEFORE_GEN,
    FD_SEEDREAM_BACKEND,
)
from .common_util import bytes_calculate_hex_md5, downscale_image_tensor
from .error_utils import ERROR_TIMEOUT, normalize_error_message
from .logging_utils import configure_default_logging
from .oss_client import upload_bytes_to_oss
from .seedream_image_size import resolution_to_seedream_size
from .webhook import webhook_send

configure_default_logging()
logger = logging.getLogger(__name__)


SEEDREAM_SUPPORTED_SIZES = ("4K", "3K", "2K", "1K")


def resolve_seedream_size(size: str) -> str:
    """校验档位并做 3K 兼容映射：豆包/image-server 只有 1K/2K/4K，3K 按 2K 请求。"""
    if size not in SEEDREAM_SUPPORTED_SIZES:
        raise ValueError(f"Invalid Seedream resolution {size!r}; expected one of {list(SEEDREAM_SUPPORTED_SIZES)}")
    if size == "3K":
        logger.warning("Seedream size=3K 不是合法档位（豆包支持 1K/2K/4K），已按 2K 请求")
        return "2K"
    return size


def build_seedream_generate_body(*, prompt: str, model: str, size: str, ratio: str, resize: bool) -> dict:
    return {
        "channel": model,
        "prompt": prompt,
        "size": resolve_seedream_size(size),
        "ratio": ratio,
        "resize": resize,
    }


def build_seedream_edit_body(
    *, prompt: str, model: str, size: str, ratio: str, resize: bool, image_urls: List[str]
) -> dict:
    body = build_seedream_generate_body(prompt=prompt, model=model, size=size, ratio=ratio, resize=resize)
    body["image_url_list"] = image_urls
    return body


class SeedreamImageClient:
    """Seedream 文生图/图生图统一客户端。

    - backend="image_generation"（默认）：文生图 POST {FD_IMAGE_GENERATE_URL}，
      图生图上传 OSS 后 POST {FD_GPT_IMAGE_EDIT_URL}，channel 区分 lite/pro 模型
    - backend="litellm"：走旧 {FD_LITELLM_BASE_URL}/v1/images/generations 直连（回退用）
    """

    DEFAULT_TIMEOUT = 300
    MAX_INPUT_PIXELS = 4096 * 4096

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        generate_url: Optional[str] = None,
        edit_url: Optional[str] = None,
        oss_uploader=None,
        request_post=None,
        request_get=None,
        webhook_name: str = "seedream",
        timeout: Optional[int] = None,
    ):
        self.backend = backend or FD_SEEDREAM_BACKEND or "image_generation"
        self.generate_url = generate_url or FD_IMAGE_GENERATE_URL
        self.edit_url = edit_url or FD_GPT_IMAGE_EDIT_URL
        self.oss_uploader = oss_uploader or upload_bytes_to_oss
        self._request_post = request_post or requests.post
        self._request_get = request_get or requests.get
        self.webhook_name = webhook_name
        self.timeout = int(timeout or self.DEFAULT_TIMEOUT)

    def generate_image(
        self, *, prompt: str, model: str, size: str, ratio: str = "1:1", resize: bool = True
    ) -> Tuple[BytesIO, str]:
        """文生图。返回 (image_bytesio, result_url)。"""
        if self.backend == "litellm":
            return self._call_litellm(prompt=prompt, model=model, size=size, ratio=ratio, image_urls=None)
        body = build_seedream_generate_body(prompt=prompt, model=model, size=size, ratio=ratio, resize=resize)
        return self._post_and_download(url=self.generate_url, body=body, label="image/generate")

    def edit_image(
        self, *, image_tensors, prompt: str, model: str, size: str, ratio: str = "1:1", resize: bool = True
    ) -> Tuple[BytesIO, str]:
        """图生图（tensor 输入，内部上传 OSS）。返回 (image_bytesio, result_url)。"""
        if self.backend == "litellm":
            return self._call_litellm(
                prompt=prompt, model=model, size=size, ratio=ratio, image_urls=self._upload_images_litellm(image_tensors)
            )
        resolve_seedream_size(size)  # 上传前先本地校验，避免无效请求浪费 OSS 流量
        image_urls = self._upload_images(image_tensors)
        body = build_seedream_edit_body(prompt=prompt, model=model, size=size, ratio=ratio, resize=resize, image_urls=image_urls)
        return self._post_and_download(url=self.edit_url, body=body, label="image/edit")

    def edit_image_with_urls(
        self, *, image_urls: List[str], prompt: str, model: str, size: str, ratio: str = "1:1", resize: bool = True
    ) -> Tuple[BytesIO, str]:
        """图生图（URL 已上传）。返回 (image_bytesio, result_url)。"""
        if self.backend == "litellm":
            return self._call_litellm(prompt=prompt, model=model, size=size, ratio=ratio, image_urls=image_urls)
        body = build_seedream_edit_body(prompt=prompt, model=model, size=size, ratio=ratio, resize=resize, image_urls=image_urls)
        return self._post_and_download(url=self.edit_url, body=body, label="image/edit")

    def _post_and_download(self, *, url: str, body: dict, label: str) -> Tuple[BytesIO, str]:
        self._notify_webhook(f"{self.webhook_name}_request", body)
        logger.info(
            "Calling image-server %s channel=%s size=%s ratio=%s resize=%s image_count=%s url=%s",
            label,
            body.get("channel"),
            body.get("size"),
            body.get("ratio"),
            body.get("resize"),
            len(body.get("image_url_list", [])),
            url,
        )
        try:
            response = self._request_post(
                url,
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            traceback.print_exc()
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail=f"{label} request timed out")
            ) from exc
        except requests.exceptions.RequestException as exc:
            traceback.print_exc()
            raise RuntimeError(normalize_error_message(f"REQUEST_ERROR: {exc}")) from exc

        if response.status_code >= 400:
            raise RuntimeError(
                normalize_error_message(f"HTTP {response.status_code} from {label}: {response.text[:500]}")
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(normalize_error_message(f"{label} 响应不是 JSON: {response.text[:500]}")) from exc

        if not isinstance(payload, dict):
            raise RuntimeError(
                normalize_error_message(f"{label} 响应格式错误: 期望 dict，实际 {type(payload).__name__}")
            )

        if payload.get("status") is not True:
            error = payload.get("error")
            message = ""
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or "")
            message = message or payload.get("message") or f"{label} 服务返回失败"
            raise RuntimeError(normalize_error_message(f"{label} 失败: {message}"))

        result_url = payload.get("result_image_url") or ""
        if not result_url:
            raise RuntimeError(normalize_error_message(f"{label} 响应缺少 result_image_url"))

        image_bytesio = self._download_result_image(result_url)
        self._notify_webhook(f"{self.webhook_name}_full", {"request": body, "response": payload})
        return image_bytesio, result_url

    def _call_litellm(
        self, *, prompt: str, model: str, size: str, ratio: str, image_urls: Optional[List[str]]
    ) -> Tuple[BytesIO, str]:
        request_size = resolution_to_seedream_size(size, ratio)
        body = {
            "model": model,
            "prompt": prompt,
            "size": request_size,
            "output_format": "png",
            "watermark": False,
        }
        if model == "doubao-seedream-5.0-lite":
            body["sequential_image_generation"] = "disabled"
        if image_urls:
            body["image"] = image_urls

        self._notify_webhook(f"{self.webhook_name}_request", body)
        logger.info("Calling Seedream API via litellm with %s", body)

        response = None
        try:
            response = self._request_post(
                url=f"{FD_LITELLM_BASE_URL}/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {FD_LITELLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            logger.info("Seedream API response: %s", result)
            result_url = result["data"][0]["url"]
            image_response = self._request_get(result_url, timeout=self.timeout)
            image_response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="request timed out")
            ) from exc
        except requests.exceptions.HTTPError as exc:
            error_response = exc.response
            status_code = error_response.status_code if error_response is not None else "unknown"
            response_text = error_response.text if error_response is not None else str(exc)
            raise RuntimeError(
                normalize_error_message(f"HTTP {status_code} from Seedream: {response_text}")
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(normalize_error_message(f"REQUEST_ERROR: {exc}")) from exc
        except Exception as exc:
            response_text = response.text[:500] if response is not None else ""
            detail = f"UNEXPECTED_ERROR: {exc}"
            if response_text:
                detail = f"{detail}; response: {response_text}"
            raise RuntimeError(normalize_error_message(detail)) from exc

        self._notify_webhook(f"{self.webhook_name}_full", {"request": body, "response": result})
        return BytesIO(image_response.content), result_url

    def _upload_images(self, image_tensors) -> List[str]:
        urls = []
        for idx, image_tensor in enumerate(image_tensors):
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            scaled = downscale_image_tensor(image_tensor, total_pixels=self.MAX_INPUT_PIXELS).squeeze(0)
            logger.info(
                "Seedream image/edit input %s resolution: input=%s scaled=%s",
                idx,
                tuple(image_tensor.squeeze(0).shape),
                tuple(scaled.shape),
            )
            image_bytes = self._tensor_to_png_bytes(scaled)
            oss_path = self._build_oss_path(image_bytes)
            url = self.oss_uploader(oss_path, image_bytes)
            urls.append(url)
            print(f"[seedream-image-edit] upload {oss_path}")
        return urls

    def _upload_images_litellm(self, image_tensors) -> List[str]:
        """litellm 回退路径保持旧行为：不降采样。"""
        urls = []
        for i in range(image_tensors.shape[0]):
            single_image = image_tensors[i : i + 1]
            scaled_image = single_image.squeeze()
            image_np = (scaled_image.numpy() * 255).astype("uint8")
            img = Image.fromarray(image_np)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{bytes_calculate_hex_md5(img_byte_arr)}.png"
            oss_file_url = self.oss_uploader(file_oss_path, img_byte_arr)
            print(f"upload {file_oss_path}")
            urls.append(oss_file_url)
        return urls

    def _build_oss_path(self, image_bytes: bytes) -> str:
        return f"{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{bytes_calculate_hex_md5(image_bytes)}.png"

    def _download_result_image(self, result_url: str) -> BytesIO:
        try:
            response = self._request_get(result_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="下载结果图超时")
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(normalize_error_message(f"REQUEST_ERROR: {exc}")) from exc
        return BytesIO(response.content)

    def _tensor_to_png_bytes(self, image_tensor) -> bytes:
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        image = Image.fromarray(arr)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _notify_webhook(self, key: str, payload: dict) -> None:
        if not FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            return
        try:
            webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {key: payload})
        except Exception:
            pass


_default_client: Optional[SeedreamImageClient] = None


def get_default_seedream_image_client() -> SeedreamImageClient:
    global _default_client
    if _default_client is None:
        _default_client = SeedreamImageClient()
    return _default_client


def reset_default_seedream_image_client() -> None:
    global _default_client
    _default_client = None
