import io

import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes.utils.gpt_image_edit_client import GptImageEditClient


def _png_bytes(color=(255, 0, 0)):
    image = Image.new("RGB", (2, 2), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class DummyResponse:
    def __init__(self, status_code=200, data=None, content=None, text=""):
        self.status_code = status_code
        self._data = data
        self.content = content or b""
        self.text = text

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_image_generation_edit_uploads_posts_and_downloads(monkeypatch):
    calls = []

    def fake_uploader(path, data):
        calls.append(("upload", path, data[:8]))
        assert path.startswith("devops/comfyui/segment_img/")
        assert path.endswith(".png")
        return "https://oss.example.com/input.png"

    def fake_post(url, headers, json, timeout):
        calls.append(("post", url, headers, json, timeout))
        assert url == "https://image-generation.example.com/image/edit"
        assert headers == {"Content-Type": "application/json", "x-request-id": "req-1"}
        assert json == {
            "channel": "gpt-image-2",
            "image_url_list": ["https://oss.example.com/input.png"],
            "prompt": "make white background",
            "size": "4K",
            "aspect_ratio": "9:16",
            "ratio": "9:16",
            "quality": "high",
        }
        return DummyResponse(data={
            "status": True,
            "result_image_url": "https://oss.example.com/result.png",
            "prompt": "make white background",
            "size": "4K",
            "cost_time": 1.2,
        })

    def fake_get(url, timeout):
        calls.append(("get", url, timeout))
        assert url == "https://oss.example.com/result.png"
        return DummyResponse(content=_png_bytes(color=(0, 255, 0)))

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.config.FD_OSS_URL_PATH_PREFIX_GPT_IMAGE", "devops/comfyui/segment_img")
    client = GptImageEditClient(
        edit_url="https://image-generation.example.com/image/edit",
        oss_uploader=fake_uploader,
        request_post=fake_post,
        request_get=fake_get,
    )

    image_bytesio, output_text, result_url = client.edit_image(
        image_tensors=[torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        prompt="make white background",
        size="4K",
        aspect_ratio="9:16",
        quality="high",
        out_request_id="req-1",
    )

    assert image_bytesio.getvalue().startswith(b"\x89PNG")
    assert output_text == "make white background"
    assert result_url == "https://oss.example.com/result.png"
    assert [call[0] for call in calls] == ["upload", "post", "get"]


def test_image_generation_edit_raises_error_message_on_status_false():
    client = GptImageEditClient(
        edit_url="https://image-generation.example.com/image/edit",
        oss_uploader=lambda path, data: "https://oss.example.com/input.png",
        request_post=lambda *args, **kwargs: DummyResponse(data={
            "status": False,
            "error": {"code": "510000", "message": "gpt-image-2服务使用出错"},
        }),
    )

    with pytest.raises(RuntimeError, match="gpt-image-2服务使用出错"):
        client.edit_image(
            image_tensors=[torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
            prompt="edit",
            size="2K",
            quality="low",
        )
