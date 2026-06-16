import io
import logging
import traceback
from typing import Optional, Tuple

import requests
from PIL import Image
from io import BytesIO

from ..config import FD_GPT_IMAGE_BACKEND, FD_GPT_IMAGE_EDIT_URL, FD_LITELLM_API_KEY, FD_LITELLM_BASE_URL
from .common_util import bytes_calculate_hex_md5, downscale_image_tensor
from .error_utils import ERROR_TIMEOUT, normalize_error_message
from .gpt_image_request import GptImageRequestMixin
from .oss_client import upload_bytes_to_oss
from .logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)


class GptImageEditClient:
    """GPT Image 图生图统一客户端。

    - backend="image_generation"（默认）：上传图片到 OSS，POST {FD_GPT_IMAGE_EDIT_URL}/image/edit
    - backend="litellm"：走旧 multipart {FD_LITELLM_BASE_URL}/v1/images/edits
    """

    DEFAULT_TIMEOUT = 300
    GPT_IMAGE_CHANNEL = "gpt-image-2"

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        edit_url: Optional[str] = None,
        oss_uploader=None,
        request_post=None,
        request_get=None,
        timeout: Optional[int] = None,
    ):
        self.backend = backend or FD_GPT_IMAGE_BACKEND or "image_generation"
        self.edit_url = edit_url or FD_GPT_IMAGE_EDIT_URL
        self.oss_uploader = oss_uploader or upload_bytes_to_oss
        self._request_post = request_post or requests.post
        self._request_get = request_get or requests.get
        self.timeout = int(timeout or self.DEFAULT_TIMEOUT)

    def edit_image(
        self,
        *,
        image_tensors,
        prompt: str,
        size: str,
        quality: str = "medium",
        out_request_id: str = "",
    ) -> Tuple[BytesIO, str, str]:
        """根据 backend 选择调用方式，返回 (image_bytesio, output_text, result_url)。"""
        if self.backend == "litellm":
            return self._edit_via_litellm(
                image_tensors=image_tensors,
                prompt=prompt,
                size=size,
                quality=quality,
                out_request_id=out_request_id,
            )
        return self._edit_via_image_generation(
            image_tensors=image_tensors,
            prompt=prompt,
            size=size,
            quality=quality,
            out_request_id=out_request_id,
        )

    def _edit_via_image_generation(
        self,
        *,
        image_tensors,
        prompt: str,
        size: str,
        quality: str,
        out_request_id: str,
    ) -> Tuple[BytesIO, str, str]:
        image_urls = self._upload_images(image_tensors)

        body = {
            "channel": self.GPT_IMAGE_CHANNEL,
            "image_url_list": image_urls,
            "prompt": prompt,
            "size": size,
            "quality": quality,
        }
        headers = {"Content-Type": "application/json"}
        if out_request_id:
            headers["x-request-id"] = out_request_id

        logger.info(
            "Calling image-generation /image/edit channel=%s size=%s quality=%s image_count=%s url=%s",
            self.GPT_IMAGE_CHANNEL, size, quality, len(image_urls), self.edit_url,
        )
        try:
            response = self._request_post(
                self.edit_url,
                headers=headers,
                json=body,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            traceback.print_exc()
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="image/edit request timed out")
            ) from exc
        except requests.exceptions.RequestException as exc:
            traceback.print_exc()
            raise RuntimeError(normalize_error_message(f"REQUEST_ERROR: {exc}")) from exc

        if response.status_code >= 400:
            raise RuntimeError(
                normalize_error_message(f"HTTP {response.status_code} from image/edit: {response.text[:500]}")
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                normalize_error_message(f"image/edit 响应不是 JSON: {response.text[:500]}")
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError(
                normalize_error_message(f"image/edit 响应格式错误: 期望 dict，实际 {type(payload).__name__}")
            )

        if payload.get("status") is not True:
            error = payload.get("error")
            message = ""
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or "")
            message = message or payload.get("message") or "image/edit 服务返回失败"
            raise RuntimeError(normalize_error_message(f"IMAGE_EDIT_FAILED: {message}"))

        result_url = payload.get("result_image_url") or ""
        if not result_url:
            raise RuntimeError(normalize_error_message("image/edit 响应缺少 result_image_url"))

        image_bytesio = self._download_result_image(result_url)
        output_text = payload.get("prompt") or ""
        return image_bytesio, output_text, result_url

    def _upload_images(self, image_tensors) -> list[str]:
        urls = []
        for idx, image_tensor in enumerate(image_tensors):
            scaled = downscale_image_tensor(image_tensor, total_pixels=4096 * 4096).squeeze(0)
            logger.info(
                "GPT image/edit input %s resolution: input=%s scaled=%s",
                idx, tuple(image_tensor.squeeze(0).shape), tuple(scaled.shape),
            )
            image_bytes = self._tensor_to_png_bytes(scaled)
            oss_path = self._build_oss_path(image_bytes)
            url = self.oss_uploader(oss_path, image_bytes)
            urls.append(url)
            print(f"[gpt-image-edit] upload {oss_path}")
        return urls

    def _build_oss_path(self, image_bytes: bytes) -> str:
        from ..config import FD_OSS_URL_PATH_PREFIX_GPT_IMAGE
        return f"{FD_OSS_URL_PATH_PREFIX_GPT_IMAGE}/{bytes_calculate_hex_md5(image_bytes)}.png"

    def _download_result_image(self, result_url: str) -> BytesIO:
        try:
            response = self._request_get(result_url, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="下载结果图超时")
            ) from exc
        response.raise_for_status()
        return BytesIO(response.content)

    def _tensor_to_png_bytes(self, image_tensor) -> bytes:
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        image = Image.fromarray(arr)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _edit_via_litellm(
        self,
        *,
        image_tensors,
        prompt: str,
        size: str,
        quality: str,
        out_request_id: str,
    ) -> Tuple[BytesIO, str, str]:
        mixin = _LiteLLMAdapter()
        multipart_files = []
        for idx, image_tensor in enumerate(image_tensors):
            scaled = downscale_image_tensor(image_tensor, total_pixels=4096 * 4096).squeeze(0)
            img_bytes = self._tensor_to_png_bytes(scaled)
            multipart_files.append(("image", (f"image_{idx}.png", img_bytes, "image/png")))

        data = {
            "model": self.GPT_IMAGE_CHANNEL,
            "prompt": prompt,
            "size": size,
            "quality": quality,
        }
        if out_request_id:
            data["user"] = out_request_id

        return mixin._call_gpt_image_with_retry_policy(
            base_url=FD_LITELLM_BASE_URL,
            api_key=FD_LITELLM_API_KEY,
            data=data,
            multipart_files=multipart_files,
            batch_size=len(multipart_files),
            logger=logger,
        )


class _LiteLLMAdapter(GptImageRequestMixin):
    pass


_default_client: Optional[GptImageEditClient] = None


def get_default_gpt_image_edit_client() -> GptImageEditClient:
    global _default_client
    if _default_client is not None:
        return _default_client
    _default_client = GptImageEditClient()
    return _default_client


def reset_default_gpt_image_edit_client() -> None:
    global _default_client
    _default_client = None
