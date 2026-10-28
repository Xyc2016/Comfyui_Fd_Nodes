"""LiteLLM 图生图三消费者和电商视觉节点的海外 URL 传图。"""

import base64
import io
import json

import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import aistudio_image_combo_node as aistudio
from src.Comfyui_Fd_Nodes import zhiyi_image_to_image_node as single
from src.Comfyui_Fd_Nodes import zhiyi_image_to_image_combo_node as combo
from src.Comfyui_Fd_Nodes.prompt_nodes import EcommercePromptGenerator
from src.Comfyui_Fd_Nodes.utils import image_payload
from src.Comfyui_Fd_Nodes.utils import litellm_gemini_image as litellm


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("FD_IMAGE_DELIVERY", raising=False)
    monkeypatch.delenv("FD_OVERSEAS_GPU_GATEWAY", raising=False)


def invoke(consumer, monkeypatch):
    image = torch.ones((1, 73, 127, 3))
    options = dict(model="gemini-3-pro-image-preview-siphonlab", aspect_ratio="3:4", image_size="2K",
                   system_prompt="系统提示", seed_mode="固定种子", seed=13)
    if consumer == "single":
        return single.ZhiYiImageToImageNode().generate(image_1=image, prompt="原提示", **options)
    if consumer == "combo":
        return combo.ZhiYiImageToImageComboNode().generate(combo_1={"images": [image], "prompts": ["原提示"]}, **options)
    return aistudio.ZhiYiAiStudioImageComboNode().generate(
        combo_1={"images": [image], "prompts": ["原提示"]}, target_url="https://chat.example.com", api_key="test", **options,
    )


@pytest.mark.parametrize("consumer", ["single", "combo", "aistudio"])
@pytest.mark.parametrize("overseas", [False, True])
def test_three_litellm_consumers_delivery(monkeypatch, consumer, overseas):
    uploads, model_calls = [], []
    if overseas:
        monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "https://gateway.example.com")
        monkeypatch.setenv("FD_IMAGE_DELIVERY", "base64")

    def upload(path, data):
        uploads.append(data)
        return "https://cdn.example.com/input.png"

    def call(**kwargs):
        model_calls.append(kwargs)
        return torch.ones((1, 2, 2, 3))

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", upload)
    for module in (single, combo, aistudio):
        monkeypatch.setattr(module, "call_litellm_gemini_image", call)
    invoke(consumer, monkeypatch)
    assert len(model_calls) == 1
    request = model_calls[0]
    assert request["model"] == "gemini-3-pro-image-preview-siphonlab"
    assert request["aspect_ratio"] == "3:4" and request["image_size"] == "2K" and request["seed"] == 13
    messages = request["messages"]
    assert messages[0]["content"] == [{"type": "text", "text": "系统提示"}]
    assert messages[1]["content"][0] == {"type": "text", "text": "原提示"}
    image_ref = messages[1]["content"][1]["image_url"]["url"]
    if overseas:
        assert image_ref == "https://cdn.example.com/input.png" and len(uploads) == 1
        assert Image.open(io.BytesIO(uploads[0])).size == (127, 73)
        assert "data:" not in json.dumps(messages) and "base64" not in json.dumps(messages)
    else:
        assert uploads == [] and image_ref.startswith("data:image/png;base64,")
        assert Image.open(io.BytesIO(base64.b64decode(image_ref.split(",", 1)[1]))).size == (127, 73)


@pytest.mark.parametrize("consumer", ["single", "combo", "aistudio"])
def test_three_consumers_upload_failure_prevents_model(monkeypatch, consumer):
    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")

    def failed(*args):
        raise RuntimeError("OSS unavailable")

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", failed)
    for module in (single, combo, aistudio):
        monkeypatch.setattr(module, "call_litellm_gemini_image", lambda **k: pytest.fail("上传失败禁止调用模型"))
    with pytest.raises(RuntimeError, match="OSS unavailable"):
        invoke(consumer, monkeypatch)


def test_build_messages_keeps_http_reference_and_rejects_overseas_base64(monkeypatch):
    url = "https://cdn.example.com/input.png"
    assert litellm.build_litellm_messages("p", [url])[0]["content"][1]["image_url"]["url"] == url
    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
    with pytest.raises(RuntimeError, match="HTTP"):
        litellm.build_litellm_messages("p", ["AAAA"])


def test_ecommerce_overseas_uploads_original_batch_and_sends_urls(monkeypatch):
    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
    images = torch.ones((2, 1100, 37, 3))
    uploads, calls = [], []

    def upload(path, data):
        uploads.append(data)
        return f"https://cdn.example.com/{len(uploads)}.png"

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def post(**kwargs):
        calls.append(json.loads(kwargs["data"]))
        return Response()

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", upload)
    monkeypatch.setattr(litellm.requests, "post", post)
    node = EcommercePromptGenerator()
    refs = node.collect_base64_images([images])
    assert refs == ["https://cdn.example.com/1.png", "https://cdn.example.com/2.png"]
    assert all(Image.open(io.BytesIO(data)).size == (37, 1100) for data in uploads)
    assert node.call_llm_vision("model", "system", "user", refs)["success"] is True
    assert "data:" not in json.dumps(calls) and "base64" not in json.dumps(calls)


def test_ecommerce_upload_failure_does_not_retry_other_batch_image_or_call_model(monkeypatch):
    monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "https://gateway.example.com")
    calls = []

    def failed(*args):
        calls.append(1)
        raise RuntimeError("OSS unavailable")

    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", failed)
    monkeypatch.setattr(litellm.requests, "post", lambda **k: pytest.fail("上传失败禁止调用模型"))
    with pytest.raises(RuntimeError, match="OSS unavailable"):
        EcommercePromptGenerator().generate_prompts_with_vision(
            "model", "product", "selling", "style", "scene", "English", 0, 1,
            product_image=torch.zeros((2, 73, 127, 3)),
        )
    assert len(calls) == 1


def test_ecommerce_domestic_keeps_jpeg_resize(monkeypatch):
    monkeypatch.setattr(image_payload, "upload_bytes_to_oss", lambda *a: pytest.fail("国内不应上传"))
    ref = EcommercePromptGenerator().tensor_to_base64(torch.zeros((1, 1100, 1100, 3)))
    assert ref.startswith("data:image/jpeg;base64,")
    image = Image.open(io.BytesIO(base64.b64decode(ref.split(",", 1)[1])))
    assert image.size == (1024, 1024) and image.format == "JPEG"
