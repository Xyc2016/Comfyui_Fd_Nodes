import base64
import hashlib
import io

import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes.utils import image_payload as payload_module


def test_base64_preserves_dimensions_and_pixels_without_upload(monkeypatch):
    def unexpected_upload(*args):
        pytest.fail("base64 模式不应上传 OSS")

    monkeypatch.setattr(payload_module, "upload_bytes_to_oss", unexpected_upload)
    tensor = torch.zeros((1, 7, 11, 3))
    tensor[..., 0] = 1
    result = payload_module.tensor_to_image_payload(tensor, "base64")
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[1])))
    assert result.startswith("data:image/png;base64,")
    assert image.size == (11, 7)
    assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("source, expected", [
    ("https://bucket.oss-cn-hangzhou.aliyuncs.com/input.png", "https://bucket.oss-cn-hangzhou-internal.aliyuncs.com/input.png"),
    ("https://cdn.example.com/input.png", "https://cdn.example.com/input.png"),
])
def test_url_uploads_original_png_with_content_hash(monkeypatch, source, expected):
    uploads = []

    def upload(path, data):
        uploads.append((path, data))
        return source

    monkeypatch.setattr(payload_module, "upload_bytes_to_oss", upload)
    monkeypatch.setattr(payload_module, "FD_OSS_URL_PATH_PREFIX_GPU_INPUT", "test/gpu/")
    tensor = torch.ones((1, 9, 13, 3))
    assert payload_module.tensor_to_image_payload(tensor, "url") == expected
    path, data = uploads[0]
    assert path == f"test/gpu/{hashlib.md5(data).hexdigest()}.png"
    image = Image.open(io.BytesIO(data))
    assert image.size == (13, 9)
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_url_upload_failure_has_configuration_context(monkeypatch):
    def upload(*args):
        raise RuntimeError("未配置 OSS 上传参数: FD_OSS_BUCKET_NAME")

    monkeypatch.setattr(payload_module, "upload_bytes_to_oss", upload)
    with pytest.raises(RuntimeError, match="image_delivery=url.*FD_OSS_.*FD_OSS_BUCKET_NAME"):
        payload_module.tensor_to_image_payload(torch.zeros((1, 2, 2, 3)), "url")


def test_unknown_delivery_is_rejected():
    with pytest.raises(RuntimeError, match="未知的 image_delivery"):
        payload_module.tensor_to_image_payload(torch.zeros((1, 2, 2, 3)), "invalid")


@pytest.mark.parametrize("key, expected", [
    (None, {}), ("  ", {}), (" secret ", {"Authorization": "Bearer secret"}),
])
def test_auth_headers(key, expected):
    assert payload_module.auth_headers(key) == expected
