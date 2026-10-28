"""GPU 图像服务的 image 字段编码与 OSS 上传。"""

import base64
import io
import os
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from ..config import FD_OSS_URL_PATH_PREFIX_GPU_INPUT
from .common_util import bytes_calculate_hex_md5
from .oss_client import upload_bytes_to_oss
from .url_utils import is_url_delivery_required, normalize_oss_url

IMAGE_DELIVERY_BASE64 = "base64"
IMAGE_DELIVERY_URL = "url"
IMAGE_DELIVERY_OPTIONS = [IMAGE_DELIVERY_BASE64, IMAGE_DELIVERY_URL]


def resolve_image_delivery(image_delivery=None) -> str:
    """显式海外部署或全局 URL 门控禁止降级；国内缺省保持 base64。

    不读取 config.FD_OVERSEAS_GPU_GATEWAY：该常量有供 UI 选项使用的默认地址。
    """
    configured = os.environ.get("FD_IMAGE_DELIVERY", "").strip().lower()
    if configured and configured not in IMAGE_DELIVERY_OPTIONS:
        raise RuntimeError(f"未知的 FD_IMAGE_DELIVERY: {configured}")
    if image_delivery is not None and image_delivery not in IMAGE_DELIVERY_OPTIONS:
        raise RuntimeError(f"未知的 image_delivery: {image_delivery}；可选值为: {', '.join(IMAGE_DELIVERY_OPTIONS)}")
    if is_url_delivery_required():
        return IMAGE_DELIVERY_URL
    return image_delivery or configured or IMAGE_DELIVERY_BASE64


def image_delivery_input_config() -> dict:
    """image_delivery 下拉的注册配置，各节点保持一致。"""
    return {
        "default": resolve_image_delivery(),
        "tooltip": (
            "图片传给服务端的方式：base64 为原方式；url 先上传 OSS 再传链接"
            "（大图/海外网关场景减少请求体，需配置 FD_OSS_*；服务端要下载图片，timeout 建议 ≥60 秒）"
        ),
    }


def auth_headers(api_key) -> dict:
    """服务端启用 API key 时返回 Authorization 头，未配置返回空 dict。"""
    key = (api_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def tensor_to_rgb_pil(image_tensor):
    """tensor → RGB PIL，与各节点的 _image_tensor_to_rgb_pil 转换逻辑一致。"""
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise RuntimeError(f"图片 tensor shape 错误: {tuple(image_tensor.shape)}")
    return Image.fromarray(arr[..., :3]).convert("RGB")


def tensor_to_png_bytes(image_tensor) -> bytes:
    image = tensor_to_rgb_pil(image_tensor)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def require_http_image_url(url) -> str:
    """上传结果必须是可供下游 GET 的 HTTP(S) 引用。"""
    if not isinstance(url, str):
        raise RuntimeError("图片上传未返回有效的 HTTP(S) URL")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or any(c.isspace() for c in url):
        raise RuntimeError("图片上传未返回有效的 HTTP(S) URL")
    return url


def _upload_png_bytes_to_oss_url(png_bytes: bytes, *, internal=True) -> str:
    object_path = f"{FD_OSS_URL_PATH_PREFIX_GPU_INPUT.rstrip('/')}/{bytes_calculate_hex_md5(png_bytes)}.png"
    try:
        url = upload_bytes_to_oss(object_path, png_bytes)
    except Exception as exc:
        raise RuntimeError(
            f"image_delivery=url 上传 OSS 失败（请检查 FD_OSS_* 配置或网络）: {exc}"
        ) from exc
    url = require_http_image_url(url)
    # 杭州 GPU 服务保留内网下载；图生文的 LiteLLM HTTP 引用使用原始公网 URL。
    return normalize_oss_url(url) if internal or is_url_delivery_required() else url


def tensor_to_image_payload(image_tensor, image_delivery=None, *, internal=True) -> str:
    """构造图片引用：国内 base64 保持 data URL，海外仅使用 OSS URL。"""
    image_delivery = resolve_image_delivery(image_delivery)
    if image_delivery == IMAGE_DELIVERY_URL:
        png_bytes = tensor_to_png_bytes(image_tensor)
        if internal:
            return _upload_png_bytes_to_oss_url(png_bytes)
        return _upload_png_bytes_to_oss_url(png_bytes, internal=False)
    if image_delivery != IMAGE_DELIVERY_BASE64:
        raise RuntimeError(
            f"未知的 image_delivery: {image_delivery}；可选值为: {', '.join(IMAGE_DELIVERY_OPTIONS)}"
        )
    encoded = base64.b64encode(tensor_to_png_bytes(image_tensor)).decode("ascii")
    return f"data:image/png;base64,{encoded}"
