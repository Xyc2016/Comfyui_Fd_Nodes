import requests
import pytest
import torch

from src.Comfyui_Fd_Nodes import nodes as nodes_module
from src.Comfyui_Fd_Nodes.nodes import FD_SeedreamImage
from src.Comfyui_Fd_Nodes.old_gemini_api_node import GenImageServiceError
from src.Comfyui_Fd_Nodes.utils.seedream_image_client import SeedreamImageClient


class OkResponse:
    def __init__(self, payload, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


def make_fake_client(*, uploaded_urls=None, post_handler=None, download_content=b"fake-image"):
    """构造注入 fake HTTP 的真实 SeedreamImageClient。"""
    if uploaded_urls is None:
        uploaded_urls = ["https://oss.example.com/input.png"]

    uploaded_iter = iter(uploaded_urls)

    def fake_upload(path, data):
        return next(uploaded_iter)

    def fake_post(url, headers, json, timeout):
        if post_handler is not None:
            return post_handler(url, headers, json, timeout)
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    class GetResponse:
        content = download_content

        def raise_for_status(self):
            return None

    client = SeedreamImageClient(
        backend="image_generation",
        generate_url="http://image-server/image/generate",
        edit_url="http://image-server/image/edit",
        oss_uploader=fake_upload,
        request_post=fake_post,
        request_get=lambda url, timeout: GetResponse(),
        timeout=60,
    )
    return client


def attach_client(monkeypatch, client):
    monkeypatch.setattr(nodes_module, "get_default_seedream_image_client", lambda: client)


def test_seedream_image_node_metadata_exposes_aspect_ratio():
    input_types = FD_SeedreamImage.INPUT_TYPES()

    assert input_types["required"]["model"][0] == ["doubao-seedream-5.0-lite", "doubao-seedream-5.0-pro"]
    assert input_types["required"]["size"][0] == ["4K", "3K", "2K", "1K"]
    assert input_types["required"]["size"][1]["default"] == "2K"
    assert input_types["optional"]["aspect_ratio"][0] == [
        "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9", "9:21",
    ]
    assert input_types["optional"]["output_format"][0] == ["png", "jpg"]


def test_seedream_image_node_generate_calls_image_server(monkeypatch):
    captured = []

    def post_handler(url, headers, json, timeout):
        captured.append((url, headers, json, timeout))
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    attach_client(monkeypatch, make_fake_client(post_handler=post_handler))
    monkeypatch.setattr(nodes_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    node = FD_SeedreamImage()
    result = node.api_call(
        prompt="test prompt",
        model="doubao-seedream-5.0-pro",
        size="4K",
        output_format="png",
        aspect_ratio="3:4",
    )

    assert result == ("fake-image",)
    url, headers, request_body, timeout = captured[0]
    assert url == "http://image-server/image/generate"
    assert headers["Content-Type"] == "application/json"
    assert timeout == 60
    assert request_body == {
        "channel": "doubao-seedream-5.0-pro",
        "prompt": "test prompt",
        "size": "4K",
        "ratio": "3:4",
        "resize": True,
    }


def test_seedream_image_node_generate_maps_3k_to_2k(monkeypatch):
    captured = []

    def post_handler(url, headers, json, timeout):
        captured.append(json)
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    attach_client(monkeypatch, make_fake_client(post_handler=post_handler))
    monkeypatch.setattr(nodes_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    node = FD_SeedreamImage()
    node.api_call("test prompt", "doubao-seedream-5.0-lite", "3K", aspect_ratio="1:1")

    assert captured[0]["size"] == "2K"


def test_seedream_image_node_edit_uploads_oss_and_posts_image_server(monkeypatch):
    uploaded_urls = ["https://oss.example.com/input-1.png", "https://oss.example.com/input-2.png"]
    captured = []

    def post_handler(url, headers, json, timeout):
        captured.append(json)
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    attach_client(monkeypatch, make_fake_client(uploaded_urls=uploaded_urls, post_handler=post_handler))
    monkeypatch.setattr(nodes_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
    node = FD_SeedreamImage()
    result = node.api_call("test prompt", "doubao-seedream-5.0-lite", "1K", images=images, aspect_ratio="9:21")

    assert result == ("fake-image",)
    assert captured[0]["channel"] == "doubao-seedream-5.0-lite"
    assert captured[0]["size"] == "1K"
    assert captured[0]["ratio"] == "9:21"
    assert captured[0]["image_url_list"] == uploaded_urls


def test_seedream_image_node_rejects_invalid_size_before_upload_or_post(monkeypatch):
    uploaded = []
    posted = []

    def fake_upload(path, data):
        uploaded.append(path)
        return "https://oss.example.com/x.png"

    def fake_post(url, headers, json, timeout):
        posted.append(json)
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    client = SeedreamImageClient(
        backend="image_generation",
        generate_url="http://image-server/image/generate",
        edit_url="http://image-server/image/edit",
        oss_uploader=fake_upload,
        request_post=fake_post,
        request_get=lambda url, timeout: OkResponse(b"x"),
    )
    attach_client(monkeypatch, client)

    node = FD_SeedreamImage()
    with pytest.raises(ValueError, match=r"Invalid Seedream resolution 'invalid'.*'1K'"):
        node.api_call(
            "test prompt",
            "doubao-seedream-5.0-lite",
            "invalid",
            images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        )

    assert uploaded == []
    assert posted == []


def test_seedream_image_node_returns_http_error_detail(monkeypatch):
    def post_handler(url, headers, json, timeout):
        return OkResponse({"error": "bad"}, status_code=400, text='{"error":"image area must be at most 4624220 pixels"}')

    attach_client(monkeypatch, make_fake_client(post_handler=post_handler))

    node = FD_SeedreamImage()
    with pytest.raises(
        GenImageServiceError,
        match=r"^UNKNOWN: HTTP 400 from image/generate: ",
    ):
        node.api_call("test prompt", "doubao-seedream-5.0-pro", "4K", aspect_ratio="3:4")


def test_seedream_image_node_only_labels_real_timeout_as_timeout(monkeypatch):
    def post_handler(url, headers, json, timeout):
        raise requests.exceptions.ReadTimeout("upstream took too long")

    attach_client(monkeypatch, make_fake_client(post_handler=post_handler))

    node = FD_SeedreamImage()
    with pytest.raises(GenImageServiceError, match=r"^TIMEOUT: upstream took too long$"):
        node.api_call("test prompt", "doubao-seedream-5.0-pro", "1K", aspect_ratio="3:4")


def test_seedream_image_node_returns_download_network_error(monkeypatch):
    def post_handler(url, headers, json, timeout):
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    class BadGetResponse:
        def raise_for_status(self):
            raise requests.exceptions.SSLError("TLS connection closed")

    client = SeedreamImageClient(
        backend="image_generation",
        generate_url="http://image-server/image/generate",
        edit_url="http://image-server/image/edit",
        oss_uploader=lambda path, data: "https://oss.example.com/x.png",
        request_post=post_handler,
        request_get=lambda url, timeout: BadGetResponse(),
    )
    attach_client(monkeypatch, client)

    node = FD_SeedreamImage()
    with pytest.raises(
        GenImageServiceError,
        match=r"^UNKNOWN: REQUEST_ERROR: TLS connection closed$",
    ):
        node.api_call("test prompt", "doubao-seedream-5.0-pro", "1K", aspect_ratio="3:4")
