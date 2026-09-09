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


@pytest.mark.parametrize("quality", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize(("model", "backend"), [
    ("gpt-image-2", "image_generation"),
    ("gpt-image-2.5", "image_generation"),
    ("gpt-image-2.5", "litellm"),
    ("gpt-image-2.5-sunburst-siphonlab", "litellm"),
    ("gpt-image-2.5-flare-siphonlab", "litellm"),
    ("gpt-image-2.5-sunburst-siphonlab", "image_generation"),
    ("gpt-image-2.5-flare-siphonlab", "image_generation"),
])
def test_image_generation_edit_uploads_posts_and_downloads(monkeypatch, model, backend, quality):
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
            "channel": model,
            "image_url_list": ["https://oss.example.com/input.png"],
            "prompt": "make white background",
            "size": "4K",
            "aspect_ratio": "9:16",
            "ratio": "9:16",
            "quality": quality,
            "resize": False,
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
        backend=backend,
        edit_url="https://image-generation.example.com/image/edit",
        oss_uploader=fake_uploader,
        request_post=fake_post,
        request_get=fake_get,
    )

    image_bytesio, output_text, result_url = client.edit_image(
        image_tensors=[torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        prompt="make white background",
        model=model,
        size="4K",
        aspect_ratio="9:16",
        quality=quality,
        resize=False,
        out_request_id="req-1",
    )

    assert image_bytesio.getvalue().startswith(b"\x89PNG")
    assert output_text == "make white background"
    assert result_url == "https://oss.example.com/result.png"
    assert [call[0] for call in calls] == ["upload", "post", "get"]


def test_image_generation_edit_sends_resize_true_by_default():
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(json)
        return DummyResponse(data={
            "status": True,
            "result_image_url": "https://oss.example.com/result.png",
        })

    client = GptImageEditClient(
        backend="image_generation",
        edit_url="https://image-generation.example.com/image/edit",
        oss_uploader=lambda path, data: "https://oss.example.com/input.png",
        request_post=fake_post,
        request_get=lambda url, timeout: DummyResponse(content=_png_bytes()),
    )

    client.edit_image(
        image_tensors=[torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        prompt="edit",
        size="2K",
    )

    assert captured["resize"] is True
    assert captured["channel"] == "gpt-image-2"


@pytest.mark.parametrize("quality", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize(("model", "backend"), [
    (None, "litellm"),
    ("gpt-image-2", "litellm"),
])
def test_litellm_edit_preserves_model_size_quality_and_images(monkeypatch, model, backend, quality):
    captured = {}
    result = (io.BytesIO(_png_bytes()), "ok", "https://example.com/result.png")

    def fake_request(self, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.utils.gpt_image_edit_client._LiteLLMAdapter._call_gpt_image_with_retry_policy",
        fake_request,
    )
    client = GptImageEditClient(backend=backend)
    images = [torch.zeros((1, 2, 2, 3)), torch.ones((1, 2, 2, 3))]
    output = client.edit_image(
        image_tensors=images,
        prompt="edit both images",
        size="1537x1025",
        quality=quality,
        out_request_id="req-model",
        **({} if model is None else {"model": model}),
    )

    assert output is result
    assert captured["data"] == {
        "model": model or "gpt-image-2",
        "prompt": "edit both images",
        "size": "1537x1025",
        "quality": quality,
        "user": "req-model",
    }
    assert captured["batch_size"] == 2
    assert len(captured["multipart_files"]) == 2
    for index, (field, (filename, data, content_type)) in enumerate(captured["multipart_files"]):
        assert field == "image"
        assert filename == f"image_{index}.png"
        assert content_type == "image/png"
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (2, 2)
            assert image.convert("RGB").getpixel((0, 0)) == (index * 255,) * 3


@pytest.mark.parametrize("model", ["gpt-image-2", "gpt-image-2.5", "gpt-image-2.5-sunburst-siphonlab", "gpt-image-2.5-flare-siphonlab"])
def test_image_generation_edit_raises_error_message_on_status_false(model):
    channels = []

    def fake_post(*args, **kwargs):
        channels.append(kwargs["json"]["channel"])
        return DummyResponse(data={
            "status": False,
            "error": {"code": "510000", "message": "gpt-image-2服务使用出错"},
        })

    client = GptImageEditClient(
        backend="image_generation",
        edit_url="https://image-generation.example.com/image/edit",
        oss_uploader=lambda path, data: "https://oss.example.com/input.png",
        request_post=fake_post,
    )

    with pytest.raises(RuntimeError, match="gpt-image-2服务使用出错"):
        client.edit_image(
            image_tensors=[torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
            prompt="edit",
            size="2K",
            quality="low",
            model=model,
        )

    assert channels == [model]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, False),
        (0, False),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
        (True, True),
        (1, True),
        ("true", True),
        ("yes", True),
        ("", True),
        (None, True),
    ],
)
def test_normalize_resize_accepts_falseish_values(value, expected):
    client = GptImageEditClient()

    assert client._normalize_resize(value) is expected
