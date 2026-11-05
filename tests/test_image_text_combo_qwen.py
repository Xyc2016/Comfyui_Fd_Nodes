"""Qwen must use Chat Completions even when the legacy Doubao proxy is configured."""

import json

import pytest
import torch

from src.Comfyui_Fd_Nodes import zhiyi_image_text_combo_node as module


@pytest.mark.parametrize("vision_url", ["", "https://legacy.example.com/image/image_to_text_doubao"])
@pytest.mark.parametrize("image_delivery", ["base64", "url"])
def test_qwen_preserves_multimodal_contract_and_bypasses_doubao(monkeypatch, vision_url, image_delivery):
    monkeypatch.delenv("FD_OVERSEAS_GPU_GATEWAY", raising=False)
    monkeypatch.delenv("FD_IMAGE_DELIVERY", raising=False)
    monkeypatch.setattr(module, "FD_DOUBAO_VISION_URL", vision_url)
    monkeypatch.setattr(module, "load_config", lambda: {"base_url": "https://chat.example.com", "api_key": "test"})
    node = module.ZhiYiImageTextComboNode()
    refs = (["data:image/jpeg;base64,RED", "data:image/jpeg;base64,BLUE"] if image_delivery == "base64"
            else ["https://images.example.com/red.png", "https://images.example.com/blue.png"])
    prepared = iter(refs)
    monkeypatch.setattr(node, "_prepare_image_reference", lambda *args: (next(prepared), {}))
    captured = []

    class Response:
        status_code = 200
        text = "ok"

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": [{"type": "text", "text": "red, blue"}]}}]}

    def post(*args, **kwargs):
        captured.append(kwargs)
        return Response()

    monkeypatch.setattr(module.requests, "post", post)
    output, log = node.generate(
        combo_1={"images": [torch.zeros((2, 16, 16, 3))], "prompts": ["Compare the two images."]},
        system_prompt="Describe colors only.", temperature=0.25, max_tokens=333,
        retry_count=0, image_delivery=image_delivery,
    )
    assert output == "red, blue"
    assert "model=qwen3.7-plus" in log
    assert len(captured) == 1
    request = captured[0]
    assert request["url"] == "https://chat.example.com/v1/chat/completions"
    body = json.loads(request["data"])
    assert body["model"] == "qwen3.7-plus"
    assert body["stream"] is False
    assert body["temperature"] == 0.25 and body["max_tokens"] == 333
    assert body["messages"] == [
        {"role": "system", "content": [{"type": "text", "text": "Describe colors only."}]},
        {"role": "user", "content": [{"type": "text", "text": "Compare the two images."},
                                    *[{"type": "image_url", "image_url": {"url": ref}} for ref in refs]]},
    ]


def test_model_widget_is_appended_after_existing_widgets():
    fields = module.ZhiYiImageTextComboNode.INPUT_TYPES()
    assert list(fields["optional"])[-1] == "model"
    assert fields["optional"]["model"] == (["qwen3.7-plus", "doubao-seed-2.0-mini"], {"default": "qwen3.7-plus"})


def test_qwen_payload_does_not_change_subsequent_default_request():
    node = module.ZhiYiImageTextComboNode()
    messages = [{"role": "user", "content": []}]
    assert node._build_request_payload(messages, 0.7, 100, "doubao-seed-2.0-mini")["model"] == "doubao-seed-2.0-mini"
    assert node._build_request_payload(messages, 0.7, 100)["model"] == "qwen3.7-plus"
