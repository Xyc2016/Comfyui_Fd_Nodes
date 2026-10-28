"""GPU 图像服务的 image 字段编码与 OSS 上传。"""

import base64
import io

import numpy as np
from PIL import Image

from ..config import FD_OSS_URL_PATH_PREFIX_GPU_INPUT
from .common_util import bytes_calculate_hex_md5
from .oss_client import upload_bytes_to_oss
from .url_utils import normalize_oss_url

IMAGE_DELIVERY_BASE64 = "base64"
IMAGE_DELIVERY_URL = "url"
IMAGE_DELIVERY_OPTIONS = [IMAGE_DELIVERY_BASE64, IMAGE_DELIVERY_URL]


def image_delivery_input_config() -> dict:
    """image_delivery 下拉的注册配置，各节点保持一致。"""
    return {
        "default": IMAGE_DELIVERY_BASE64,
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


def _upload_png_bytes_to_oss_url(png_bytes: bytes) -> str:
    object_path = f"{FD_OSS_URL_PATH_PREFIX_GPU_INPUT.rstrip('/')}/{bytes_calculate_hex_md5(png_bytes)}.png"
    try:
        url = upload_bytes_to_oss(object_path, png_bytes)
    except Exception as exc:
        raise RuntimeError(
            f"image_delivery=url 上传 OSS 失败（请检查 FD_OSS_* 配置或网络）: {exc}"
        ) from exc
    # 服务端在杭州集群内，走内网 endpoint 下载；自定义域名前缀时原样返回
    return normalize_oss_url(url)


def tensor_to_image_payload(image_tensor, image_delivery) -> str:
    """构造 payload 的 image 字段值：base64 → data URL；url → OSS 内网链接。"""
    if image_delivery == IMAGE_DELIVERY_URL:
        return _upload_png_bytes_to_oss_url(tensor_to_png_bytes(image_tensor))
    if image_delivery != IMAGE_DELIVERY_BASE64:
        raise RuntimeError(
            f"未知的 image_delivery: {image_delivery}；可选值为: {', '.join(IMAGE_DELIVERY_OPTIONS)}"
        )
    encoded = base64.b64encode(tensor_to_png_bytes(image_tensor)).decode("ascii")
    return f"data:image/png;base64,{encoded}"
