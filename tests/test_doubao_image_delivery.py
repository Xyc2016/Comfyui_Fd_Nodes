"""图生文的单图/Combo、image-generation/chat 路由传图契约。"""

import base64
import io
import json

import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import old_fd_nodes
from src.Comfyui_Fd_Nodes import zhiyi_image_text_node as single_module
from src.Comfyui_Fd_Nodes import zhiyi_image_text_combo_node as combo_module
from src.Comfyui_Fd_Nodes.utils import image_payload


@pytest.fixture(autouse=True)
def domestic_environment(monkeypatch):
    monkeypatch.delenv("FD_OVERSEAS_GPU_GATEWAY", raising=False)
    monkeypatch.delenv("FD_IMAGE_DELIVERY", raising=False)


class Response:
    ok = True
    status_code = 200
    text = "ok"

    def json(self):
        return {"status": True, "result_text": "描述", "choices": [{"message": {"content": "描述"}}]}

    def raise_for_status(self):
        pass


@pytest.fixture
def request_capture(monkeypatch):
    requests = []

    def post(*args, **kwargs):
        requests.append(kwargs.get("json") or json.loads(kwargs["data"]))
        return Response()

    monkeypatch.setattr(single_module.requests, "post", post)
    for module in (single_module, combo_module):
        monkeypatch.setattr(module, "load_config", lambda: {"base_url": "https://chat.example.com", "api_key": "test"})
    return requests


def run_node(monkeypatch, route, combo, **kwargs):
    vision_url = "https://image.example.com/image/image_to_text_doubao" if route == "image-generation" else ""
    monkeypatch.setattr(single_module, "FD_DOUBAO_VISION_URL", vision_url)
    monkeypatch.setattr(combo_module, "FD_DOUBAO_VISION_URL", vision_url)
    images = torch.zeros((2 if combo else 1, 73, 127, 3))
    images[0, ..., 0] = 1
    if combo:
        images[1, ..., 1] = 1
        result = combo_module.ZhiYiImageTextComboNode().generate(
            combo_1={"images": [images], "prompt": "保持提示"},
            system_prompt="系统提示", temperature=0.35, max_tokens=77, retry_count=0, **kwargs,
        )
    else:
        result = single_module.ZhiYiImageTextNode().generate(
            images, "保持提示", node_switch=0, system_prompt="系统提示",
            temperature=0.35, max_tokens=77, **kwargs,
        )
    assert result[0] == "描述"


@pytest.mark.parametrize("gateway, setting, explicit, expected", [
    ("", "", None, "base64"),
    ("  ", "", None, "base64"),
    ("", "base64", None, "base64"),
    ("", "", "url", "url"),
    ("", "url", "base64", "url"),
    ("https://gateway.example.com", "", None, "url"),
    ("https://gateway.example.com", "base64", "base64", "url"),
])
def test_delivery_environment_matrix(monkeypatch, gateway, setting, explicit, expected):
    monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", gateway)
    monkeypatch.setenv("FD_IMAGE_DELIVERY", setting)
    assert image_payload.resolve_image_delivery(explicit) == expected
    assert image_payload.image_delivery_input_config()["default"] == image_payload.resolve_image_delivery()


@pytest.mark.parametrize("route", ["image-generation", "chat"])
@pytest.mark.parametrize("combo", [False, True])
@pytest.mark.parametrize("mode", ["explicit", "overseas-default", "overseas-no-downgrade", "env-url"])
def test_url_routes_preserve_images_and_prompts(monkeypatch, request_capture, route, combo, mode):
    uploads = []

    def upload(path, data):
        uploads.append((path, data))
        return f"https://bucket.oss-cn-hangzhou.aliyuncs.com/input-{len(uploads)}.png"

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", upload)
    kwargs = {}
    if mode == "explicit":
        kwargs["image_delivery"] = "url"
    elif mode == "env-url":
        monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
        kwargs["image_delivery"] = "base64"
    else:
        monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "https://gateway.example.com")
        if mode == "overseas-no-downgrade":
            monkeypatch.setenv("FD_IMAGE_DELIVERY", "base64")
            kwargs["image_delivery"] = "base64"
    monkeypatch.setattr(single_module.ZhiYiImageTextNode, "_image_tensor_to_data_url",
                        lambda *a, **k: pytest.fail("URL 路径禁止调用 base64/缩放逻辑"))
    run_node(monkeypatch, route, combo, **kwargs)
    assert len(request_capture) == 1
    body = request_capture[0]
    expected_urls = [f"https://bucket.oss-cn-hangzhou.aliyuncs.com/input-{i + 1}.png" for i in range(2 if combo else 1)]
    if route == "image-generation":
        assert body == {"image_url_list" if combo else "image_url": expected_urls if combo else expected_urls[0],
                        "text_prompt": "系统提示\n保持提示"}
    else:
        assert body["model"] == "doubao-seed-2.0-mini"
        assert body["temperature"] == 0.35 and body["max_tokens"] == 77 and body["stream"] is False
        assert body["messages"][0]["content"] == [{"type": "text", "text": "系统提示"}]
        content = body["messages"][1]["content"]
        assert content[0] == {"type": "text", "text": "保持提示"}
        assert [part["image_url"]["url"] for part in content[1:]] == expected_urls
    assert "data:" not in json.dumps(body) and "base64" not in json.dumps(body)
    assert len(uploads) == len(expected_urls)
    for index, (_path, data) in enumerate(uploads):
        image = Image.open(io.BytesIO(data))
        assert image.size == (127, 73)
        assert image.getpixel((0, 0)) == ((255, 0, 0) if index == 0 else (0, 255, 0))


@pytest.mark.parametrize("route", ["image-generation", "chat"])
@pytest.mark.parametrize("combo", [False, True])
@pytest.mark.parametrize("failure", ["exception", "empty", "data", "relative"])
def test_upload_failure_never_calls_model(monkeypatch, request_capture, route, combo, failure):
    monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "https://gateway.example.com")

    def upload(*args):
        if failure == "exception":
            raise RuntimeError("OSS unavailable")
        return {"empty": "", "data": "data:image/png;base64,AAAA", "relative": "/input.png"}[failure]

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", upload)
    with pytest.raises(RuntimeError, match="OSS|HTTP"):
        run_node(monkeypatch, route, combo)
    assert request_capture == []


@pytest.mark.parametrize("route", ["image-generation", "chat"])
@pytest.mark.parametrize("combo", [False, True])
def test_domestic_default_keeps_existing_base64_contract(monkeypatch, request_capture, route, combo):
    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", lambda *a: pytest.fail("国内缺省不应上传"))
    run_node(monkeypatch, route, combo)
    body = request_capture[0]
    if route == "image-generation":
        key = "image_base64_list" if combo else "image_base64"
        refs = body[key] if combo else [body[key]]
        assert body["text_prompt"] == "系统提示\n保持提示"
    else:
        refs = [part["image_url"]["url"].split(",", 1)[1] for part in body["messages"][-1]["content"][1:]]
    assert len(refs) == (2 if combo else 1)
    for ref in refs:
        image = Image.open(io.BytesIO(base64.b64decode(ref)))
        assert image.format == "JPEG" and image.size == (127, 73)


def test_legacy_doubao_already_sends_http_reference(monkeypatch):
    sent = []

    def post(*args, **kwargs):
        sent.append(json.loads(kwargs["data"]))
        response = Response()
        response.content = json.dumps(response.json()).encode()
        return response

    monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "https://gateway.example.com")
    monkeypatch.setattr(old_fd_nodes.requests, "post", post)
    url = "https://cdn.example.com/original.png"
    assert old_fd_nodes.FD_imgToText_Doubao().gen(url, "保持提示", "fallback") == ("描述",)
    assert sent[0]["image_url"] == url and sent[0]["text_prompt"] == "保持提示"
    assert "data:" not in json.dumps(sent[0]) and "base64" not in json.dumps(sent[0])


@pytest.mark.parametrize("reference", ["", "AAAA", "data:image/png;base64,AAAA"])
def test_legacy_overseas_rejects_non_http_before_model(monkeypatch, request_capture, reference):
    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
    with pytest.raises(RuntimeError, match="HTTP"):
        old_fd_nodes.FD_imgToText_Doubao().gen(reference, "提示", "fallback")
    assert request_capture == []


def test_overseas_public_upload_reference_even_when_prefix_is_internal(monkeypatch):
    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
    monkeypatch.setattr(image_payload, "upload_bytes_to_oss",
                        lambda *a: "https://bucket.oss-cn-hangzhou-internal.aliyuncs.com/a.png?signature=retained")
    ref = image_payload.tensor_to_image_payload(torch.zeros((1, 2, 2, 3)), internal=False)
    assert ref == "https://bucket.oss-cn-hangzhou.aliyuncs.com/a.png?signature=retained"


def test_legacy_upload_overseas_preserves_dimensions_and_propagates_failure(monkeypatch):
    monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "https://gateway.example.com")
    uploads = []

    def upload(path, data):
        uploads.append(data)
        return "https://cdn.example.com/a.png"

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", upload)
    assert old_fd_nodes.FD_Upload().gen(torch.ones((1, 73, 127, 3))) == ("https://cdn.example.com/a.png",)
    assert Image.open(io.BytesIO(uploads[0])).size == (127, 73)

    def failed_upload(*args):
        raise RuntimeError("OSS unavailable")

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", failed_upload)
    with pytest.raises(RuntimeError, match="OSS unavailable"):
        old_fd_nodes.FD_Upload().gen(torch.ones((1, 73, 127, 3)))
