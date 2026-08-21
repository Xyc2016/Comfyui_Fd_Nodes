import io

import pytest
import torch

from src.Comfyui_Fd_Nodes.utils.seedream_image_client import (
    SeedreamImageClient,
    build_seedream_edit_body,
    build_seedream_generate_body,
    resolve_seedream_size,
)


def make_client(**overrides):
    defaults = {
        "backend": "image_generation",
        "generate_url": "http://image-server/image/generate",
        "edit_url": "http://image-server/image/edit",
        "timeout": 60,
    }
    defaults.update(overrides)
    return SeedreamImageClient(**defaults)


class OkResponse:
    def __init__(self, payload, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


def test_generate_image_posts_image_server_body_and_downloads(monkeypatch):
    captured = {}
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(url)
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        captured["timeout"] = timeout
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    def fake_get(url, timeout):
        calls.append(url)
        return OkResponse(b"fake-image")

    client = make_client(request_post=fake_post, request_get=fake_get)
    image_bytesio, result_url = client.generate_image(
        prompt="test prompt", model="doubao-seedream-5.0-pro", size="4K", ratio="3:4", resize=True
    )

    assert captured["url"] == "http://image-server/image/generate"
    assert captured["body"] == {
        "channel": "doubao-seedream-5.0-pro",
        "prompt": "test prompt",
        "size": "4K",
        "ratio": "3:4",
        "resize": True,
    }
    assert captured["timeout"] == 60
    assert calls == ["http://image-server/image/generate", "https://example.com/result.png"]
    assert result_url == "https://example.com/result.png"
    assert image_bytesio.getvalue() == b"fake-image"


def test_generate_image_maps_3k_to_2k():
    captured = []

    def fake_post(url, headers, json, timeout):
        captured.append(json)
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    client = make_client(request_post=fake_post, request_get=lambda url, timeout: OkResponse(b"x"))

    client.generate_image(prompt="p", model="doubao-seedream-5.0-lite", size="3K", ratio="1:1")

    assert captured[0]["size"] == "2K"


def test_edit_image_uploads_oss_then_posts_with_image_url_list(monkeypatch):
    uploaded = []

    def fake_upload(path, data):
        uploaded.append(path)
        return f"https://oss.example.com/{path.split('/')[-1]}"

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["body"] = json
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    client = make_client(oss_uploader=fake_upload, request_post=fake_post, request_get=lambda url, timeout: OkResponse(b"x"))

    images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
    _, _ = client.edit_image(image_tensors=images, prompt="edit prompt", model="doubao-seedream-5.0-pro", size="2K", ratio="3:4")

    assert len(uploaded) == 2
    assert captured["body"] == {
        "channel": "doubao-seedream-5.0-pro",
        "prompt": "edit prompt",
        "size": "2K",
        "ratio": "3:4",
        "resize": True,
        "image_url_list": ["https://oss.example.com/1.png", "https://oss.example.com/2.png"],
    }


def test_edit_image_with_urls_skips_upload():
    uploaded = []

    def fake_upload(path, data):
        uploaded.append(path)
        return "https://oss.example.com/x.png"

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["body"] = json
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    client = make_client(oss_uploader=fake_upload, request_post=fake_post, request_get=lambda url, timeout: OkResponse(b"x"))

    client.edit_image_with_urls(
        image_urls=["https://oss.example.com/a.png"], prompt="p", model="doubao-seedream-5.0-lite", size="1K", ratio="9:21"
    )

    assert uploaded == []
    assert captured["body"]["image_url_list"] == ["https://oss.example.com/a.png"]


def test_edit_image_accepts_three_dim_tensor_from_batch_iteration():
    """回归：FD_SeedreamImage 直接传 4 维 tensor，enumerate 迭代出 3 维元素，
    修复前会在 downscale_image_tensor 的 shape[3] 处抛 IndexError: tuple index out of range。"""
    captured = []

    def fake_post(url, headers, json, timeout):
        captured.append(json)
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    client = make_client(request_post=fake_post, request_get=lambda url, timeout: OkResponse(b"x"))

    images = [torch.zeros((2, 2, 3), dtype=torch.float32)]  # 3 维 (H, W, C)
    _, _ = client.edit_image(image_tensors=images, prompt="p", model="m", size="2K", ratio="3:4")

    assert len(captured[0]["image_url_list"]) == 1


def test_invalid_size_raises_before_upload_or_post():
    uploaded = []
    posted = []

    def fake_upload(path, data):
        uploaded.append(path)
        return "https://oss.example.com/x.png"

    client = make_client(oss_uploader=fake_upload, request_post=lambda *a, **k: posted.append(a))

    with pytest.raises(ValueError, match=r"Invalid Seedream resolution 'invalid'.*'1K'"):
        client.edit_image(image_tensors=torch.zeros((1, 2, 2, 3), dtype=torch.float32), prompt="p", model="m", size="invalid")

    assert uploaded == []
    assert posted == []


def test_response_status_false_raises_with_error_message():
    captured = []

    def fake_post(url, headers, json, timeout):
        captured.append(json)
        return OkResponse({"status": False, "error": {"message": "size not valid"}})

    client = make_client(request_post=fake_post, request_get=lambda url, timeout: OkResponse(b"x"))

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: image/generate 失败: size not valid$"):
        client.generate_image(prompt="p", model="m", size="2K")


def test_response_missing_result_url_raises():
    def fake_post(url, headers, json, timeout):
        return OkResponse({"status": True})

    client = make_client(request_post=fake_post, request_get=lambda url, timeout: OkResponse(b"x"))

    with pytest.raises(RuntimeError, match=r"image/edit 响应缺少 result_image_url"):
        client.edit_image_with_urls(image_urls=["u"], prompt="p", model="m", size="2K")


def test_http_400_raises_normalized_error():
    def fake_post(url, headers, json, timeout):
        resp = OkResponse({"error": "bad"}, status_code=400, text='{"error":"image area must be at most 4624220 pixels"}')
        return resp

    client = make_client(request_post=fake_post, request_get=lambda url, timeout: OkResponse(b"x"))

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: HTTP 400 from image/generate: "):
        client.generate_image(prompt="p", model="m", size="4K")


def test_litellm_backend_keeps_legacy_body_and_pixel_size():
    captured = []

    class LitellmOkResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"url": "https://example.com/result.png"}]}

    class LitellmGetResponse:
        content = b"fake-image"

        def raise_for_status(self):
            return None

    def fake_post(url, headers, json, timeout):
        captured.append((url, headers, json, timeout))
        return LitellmOkResponse()

    client = make_client(
        backend="litellm",
        request_post=fake_post,
        request_get=lambda url, timeout: LitellmGetResponse(),
    )
    client.generate_image(prompt="p", model="doubao-seedream-5.0-pro", size="2K", ratio="3:4")

    url, headers, body, timeout = captured[0]
    assert url.endswith("/v1/images/generations")
    assert headers["Authorization"].startswith("Bearer ")
    assert timeout == 60
    assert body == {
        "model": "doubao-seedream-5.0-pro",
        "prompt": "p",
        "size": "1728x2304",
        "output_format": "png",
        "watermark": False,
    }


def test_litellm_backend_keeps_3k_pixel_conversion():
    captured = []

    class LitellmOkResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"url": "https://example.com/result.png"}]}

    class LitellmGetResponse:
        content = b"fake-image"

        def raise_for_status(self):
            return None

    def fake_post(url, headers, json, timeout):
        captured.append(json)
        return LitellmOkResponse()

    client = make_client(backend="litellm", request_post=fake_post, request_get=lambda url, timeout: LitellmGetResponse())
    client.generate_image(prompt="p", model="doubao-seedream-5.0-pro", size="3K", ratio="1:1")

    assert captured[0]["size"] == "3072x3072"


def test_litellm_backend_wraps_http_error_message():
    import requests

    response = requests.Response()
    response.status_code = 422
    response._content = b'{"error":"unsupported size 864x1152"}'

    def fake_post(url, headers, json, timeout):
        raise requests.exceptions.HTTPError(response=response)

    client = make_client(backend="litellm", request_post=fake_post, request_get=lambda url, timeout: None)

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: HTTP 422 from Seedream: "):
        client.generate_image(prompt="p", model="doubao-seedream-5.0-pro", size="1K", ratio="3:4")


def test_build_helpers_include_expected_fields():
    body = build_seedream_generate_body(prompt="p", model="m", size="2K", ratio="16:9", resize=False)
    assert body == {"channel": "m", "prompt": "p", "size": "2K", "ratio": "16:9", "resize": False}

    edit_body = build_seedream_edit_body(prompt="p", model="m", size="1K", ratio="1:1", resize=True, image_urls=["u"])
    assert edit_body["image_url_list"] == ["u"]


def test_resolve_seedream_size_maps_3k_and_rejects_unknown():
    assert resolve_seedream_size("3K") == "2K"
    assert resolve_seedream_size("4K") == "4K"
    with pytest.raises(ValueError):
        resolve_seedream_size("5K")
