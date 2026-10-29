import io

import pytest
import requests
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes.nodes import FD_RemoveWatermark
from src.Comfyui_Fd_Nodes.utils import image_payload


def call_node(image: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor]:
    params = {
        "service_url": "http://gateway/gpu/online-server-gray/online-remove-watermark/v1/process",
        "text_prompt": "watermark text.", "threshold": 0.3, "text_threshold": 0.35,
        "max_side": 3000, "mask_dilate_ksize": 9, "mask_dilate_iters": 2,
        "inpaint_method": "lama", "enable_quality_check": True,
        "fallback_to_original": True, "jpeg_quality": 95,
    }
    params.update(kwargs)
    return FD_RemoveWatermark().remove_watermark(image, **params)


@pytest.fixture
def image() -> torch.Tensor:
    return torch.full((1, 8, 10, 3), 0.5)


def test_overseas_uploads_png_and_sends_only_url_form(
    monkeypatch: pytest.MonkeyPatch, image: torch.Tensor,
) -> None:
    monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "http://gateway/gpu/online-server-gray")
    image_url = "https://cdn.example.com/xyc_dev/watermark/input.png"

    def upload(path: str, data: bytes) -> str:
        assert path.endswith(".png")
        assert Image.open(io.BytesIO(data)).size == (10, 8)
        return image_url

    def post(url: str, *, data: dict, timeout: int) -> requests.Response:
        assert data["image_url"] == image_url
        assert "image" not in data
        assert data["fallback_to_original"] is True
        assert timeout == 180
        response = requests.Response()
        response.status_code = 200
        content = io.BytesIO()
        Image.new("RGB", (10, 8), "red").save(content, format="PNG")
        response._content = content.getvalue()
        return response

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", upload)
    monkeypatch.setattr(requests, "post", post)
    result = call_node(image, image_delivery="multipart")[0]
    assert result.shape == image.shape
    assert result[0, 0, 0].tolist() == [1.0, 0.0, 0.0]


def test_domestic_keeps_multipart(monkeypatch: pytest.MonkeyPatch, image: torch.Tensor) -> None:
    monkeypatch.delenv("FD_OVERSEAS_GPU_GATEWAY", raising=False)
    monkeypatch.delenv("FD_IMAGE_DELIVERY", raising=False)

    def post(url: str, *, files: dict, data: dict, timeout: int) -> requests.Response:
        assert "image_url" not in data
        assert timeout == 60
        response = requests.Response()
        response.status_code = 200
        response._content = files["image"][1].getvalue()
        return response

    monkeypatch.setattr(requests, "post", post)
    assert call_node(image)[0].shape == image.shape


@pytest.mark.parametrize("stage", ["upload", "http", "decode"])
def test_url_failures_never_return_original(
    monkeypatch: pytest.MonkeyPatch, image: torch.Tensor, stage: str,
) -> None:
    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")

    def upload(path: str, data: bytes) -> str:
        if stage == "upload":
            raise requests.ConnectionError("upload failed")
        return "https://cdn.example.com/xyc_dev/input.png"

    def post(*args: object, **kwargs: object) -> requests.Response:
        response = requests.Response()
        response.status_code = 502 if stage == "http" else 200
        response._content = b"invalid image"
        return response

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", upload)
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RuntimeError, match="URL request failed"):
        call_node(image)
