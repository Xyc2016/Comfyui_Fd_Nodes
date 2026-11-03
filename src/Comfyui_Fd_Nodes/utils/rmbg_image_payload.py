"""JPEG Q95 inference copies for RMBG; the source tensor stays unchanged."""

import base64
import io

import torch

from ..config import FD_OSS_URL_PATH_PREFIX_GPU_INPUT
from .common_util import bytes_calculate_hex_md5
from .image_payload import IMAGE_DELIVERY_URL, require_http_image_url, resolve_image_delivery, tensor_to_rgb_pil
from .oss_client import upload_bytes_to_oss
from .url_utils import normalize_oss_url

def tensor_to_jpeg_q95_bytes(image_tensor: torch.Tensor) -> bytes:
    image = tensor_to_rgb_pil(image_tensor)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, optimize=True, subsampling=2, progressive=False)
    return buffer.getvalue()


def tensor_to_jpeg_q95_payload(image_tensor: torch.Tensor, image_delivery: str) -> str:
    delivery = resolve_image_delivery(image_delivery)
    data = tensor_to_jpeg_q95_bytes(image_tensor)
    if delivery != IMAGE_DELIVERY_URL:
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    object_path = f"{FD_OSS_URL_PATH_PREFIX_GPU_INPUT.rstrip('/')}/{bytes_calculate_hex_md5(data)}.jpg"
    try:
        url = upload_bytes_to_oss(object_path, data)
    except Exception as exc:
        raise RuntimeError(f"RMBG JPEG Q95 上传 OSS 失败（请检查 FD_OSS_* 配置或网络）: {exc}") from exc
    return normalize_oss_url(require_http_image_url(url))
