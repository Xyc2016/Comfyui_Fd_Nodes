from src.Comfyui_Fd_Nodes import nodes as nodes_module
from src.Comfyui_Fd_Nodes.nodes import FD_SeedreamImage


def test_seedream_image_node_metadata_exposes_aspect_ratio():
    input_types = FD_SeedreamImage.INPUT_TYPES()

    assert input_types["required"]["size"][0] == ["4K", "3K", "2K"]
    assert "1K" not in input_types["required"]["size"][0]
    assert input_types["optional"]["aspect_ratio"][0] == ["1:1", "3:4", "4:3", "16:9", "9:16", "3:2", "2:3", "21:9"]


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
    assert "aspect_ratio" not in request_body
    assert "ratio" not in request_body
    assert captured[1] == ("get", "https://example.com/result.png")
