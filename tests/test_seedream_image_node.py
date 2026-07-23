import pytest
import torch

from src.Comfyui_Fd_Nodes import nodes as nodes_module
from src.Comfyui_Fd_Nodes.nodes import FD_SeedreamImage


def test_seedream_image_node_metadata_exposes_aspect_ratio():
    input_types = FD_SeedreamImage.INPUT_TYPES()

    assert input_types["required"]["model"][0] == ["doubao-seedream-5.0-lite", "doubao-seedream-5.0-pro"]
    assert input_types["required"]["size"][0] == ["4K", "3K", "2K", "1K"]
    assert input_types["required"]["size"][1]["default"] == "2K"
    assert input_types["optional"]["aspect_ratio"][0] == ["1:1", "3:4", "4:3", "16:9", "9:16", "3:2", "2:3", "21:9"]
    assert input_types["optional"]["output_format"][0] == ["png", "jpg"]


def test_seedream_image_node_sends_pixel_size_without_ratio_fields(monkeypatch):
    node = FD_SeedreamImage()
    captured = []

    class DummyPostResponse:
        status_code = 200
        content = b'{"data":[{"url":"https://example.com/result.png"}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"url": "https://example.com/result.png"}]}

    class DummyGetResponse:
        content = b"fake-image"

    def fake_post(url, headers, json, timeout):
        captured.append(("post", url, headers, json, timeout))
        return DummyPostResponse()

    def fake_get(url):
        captured.append(("get", url))
        return DummyGetResponse()

    monkeypatch.setattr(nodes_module.requests, "post", fake_post)
    monkeypatch.setattr(nodes_module.requests, "get", fake_get)
    monkeypatch.setattr(nodes_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    result = node.api_call(
        prompt="test prompt",
        model="doubao-seedream-5.0-lite",
        size="4K",
        output_format="png",
        aspect_ratio="21:9",
    )

    assert result == ("fake-image",)
    _, url, headers, request_body, timeout = captured[0]
    assert url.endswith("/v1/images/generations")
    assert headers["Content-Type"] == "application/json"
    assert timeout == 300
    assert request_body["size"] == "6240x2656"
    assert request_body["sequential_image_generation"] == "disabled"
    assert "aspect_ratio" not in request_body
    assert "ratio" not in request_body
    assert captured[1] == ("get", "https://example.com/result.png")


def test_seedream_image_node_omits_sequential_image_generation_for_pro(monkeypatch):
    node = FD_SeedreamImage()
    captured = []

    class DummyPostResponse:
        status_code = 200
        content = b'{"data":[{"url":"https://example.com/result.png"}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"url": "https://example.com/result.png"}]}

    class DummyGetResponse:
        content = b"fake-image"

    def fake_post(url, headers, json, timeout):
        captured.append(("post", url, headers, json, timeout))
        return DummyPostResponse()

    monkeypatch.setattr(nodes_module.requests, "post", fake_post)
    monkeypatch.setattr(nodes_module.requests, "get", lambda url: DummyGetResponse())
    monkeypatch.setattr(nodes_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    result = node.api_call(
        prompt="test prompt",
        model="doubao-seedream-5.0-pro",
        size="2K",
        output_format="png",
        aspect_ratio="1:1",
    )

    assert result == ("fake-image",)
    request_body = captured[0][3]
    assert request_body["model"] == "doubao-seedream-5.0-pro"
    assert request_body["size"] == "2048x2048"
    assert "sequential_image_generation" not in request_body


@pytest.mark.parametrize(
    ("model", "has_sequential_field"),
    [
        ("doubao-seedream-5.0-lite", True),
        ("doubao-seedream-5.0-pro", False),
    ],
)
def test_seedream_image_node_sends_1k_pixel_size_for_lite_and_pro(monkeypatch, model, has_sequential_field):
    node = FD_SeedreamImage()
    captured = []

    class DummyPostResponse:
        status_code = 200
        content = b'{"data":[{"url":"https://example.com/result.png"}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"url": "https://example.com/result.png"}]}

    class DummyGetResponse:
        content = b"fake-image"

    def fake_post(url, headers, json, timeout):
        captured.append(json)
        return DummyPostResponse()

    monkeypatch.setattr(nodes_module.requests, "post", fake_post)
    monkeypatch.setattr(nodes_module.requests, "get", lambda url: DummyGetResponse())
    monkeypatch.setattr(nodes_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    assert node.api_call("test prompt", model, "1K", aspect_ratio="1:1") == ("fake-image",)
    assert captured[0]["size"] == "1024x1024"
    assert ("sequential_image_generation" in captured[0]) is has_sequential_field


def test_seedream_image_node_keeps_two_image_oss_url_order_for_1k(monkeypatch):
    node = FD_SeedreamImage()
    uploaded_urls = iter(["https://example.com/input-1.png", "https://example.com/input-2.png"])
    captured = []

    class DummyPostResponse:
        status_code = 200
        content = b'{"data":[{"url":"https://example.com/result.png"}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"url": "https://example.com/result.png"}]}

    class DummyGetResponse:
        content = b"fake-image"

    monkeypatch.setattr(nodes_module, "upload_bytes_to_oss", lambda path, data: next(uploaded_urls))
    monkeypatch.setattr(nodes_module.requests, "post", lambda url, headers, json, timeout: captured.append(json) or DummyPostResponse())
    monkeypatch.setattr(nodes_module.requests, "get", lambda url: DummyGetResponse())
    monkeypatch.setattr(nodes_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
    assert node.api_call("test prompt", "doubao-seedream-5.0-lite", "1K", images=images) == ("fake-image",)
    assert captured[0]["size"] == "1024x1024"
    assert captured[0]["image"] == [
        "https://example.com/input-1.png",
        "https://example.com/input-2.png",
    ]
    assert captured[0]["sequential_image_generation"] == "disabled"


def test_seedream_image_node_rejects_invalid_size_before_upload_or_post(monkeypatch):
    node = FD_SeedreamImage()
    calls = []
    monkeypatch.setattr(nodes_module, "upload_bytes_to_oss", lambda path, data: calls.append("upload"))
    monkeypatch.setattr(nodes_module.requests, "post", lambda **kwargs: calls.append("post"))

    with pytest.raises(ValueError, match=r"Invalid Seedream resolution 'invalid'.*'1K'"):
        node.api_call(
            "test prompt",
            "doubao-seedream-5.0-lite",
            "invalid",
            images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        )

    assert calls == []
