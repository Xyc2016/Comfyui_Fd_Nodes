"""Qwen must use Chat Completions even when the legacy Doubao proxy is configured."""

import json

import pytest
import torch

from src.Comfyui_Fd_Nodes import zhiyi_image_text_combo_node as module


@pytest.mark.parametrize("vision_url", ["", "https://legacy.example.com/image/image_to_text_doubao"])
@pytest.mark.parametrize("image_delivery", [None, "base64", "url"])
@pytest.mark.parametrize("overseas", [False, True])
def test_qwen_preserves_multimodal_contract_and_bypasses_doubao(
    monkeypatch: pytest.MonkeyPatch, vision_url: str, image_delivery: str | None, overseas: bool,
) -> None:
    monkeypatch.delenv("FD_OVERSEAS_GPU_GATEWAY", raising=False)
    monkeypatch.delenv("FD_IMAGE_DELIVERY", raising=False)
    if overseas:
        monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "https://gateway.example.com")
        monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
    monkeypatch.setattr(module, "FD_DOUBAO_VISION_URL", vision_url)
    monkeypatch.setattr(module, "load_config", lambda: {"base_url": "https://chat.example.com", "api_key": "test"})
    node = module.ZhiYiImageTextComboNode()
    expected_delivery = "base64"
    refs = ["data:image/jpeg;base64,RED", "data:image/jpeg;base64,BLUE"]
    prepared = iter(refs)
    def prepare(_image: torch.Tensor, delivery: str, _budget: int) -> tuple[str, dict[str, object]]:
        assert delivery == expected_delivery
        return next(prepared), {}

    monkeypatch.setattr(node, "_prepare_image_reference", prepare)
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
    assert body["enable_thinking"] is False
    assert body["stream"] is False
    assert body["temperature"] == 0.25 and body["max_tokens"] == 333
    assert body["messages"] == [
        {"role": "system", "content": [{"type": "text", "text": "Describe colors only."}]},
        {"role": "user", "content": [{"type": "text", "text": "Compare the two images."},
                                    *[{"type": "image_url", "image_url": {"url": ref}} for ref in refs]]},
    ]


def test_model_widget_is_appended_after_existing_widgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", "https://gateway.example.com")
    fields = module.ZhiYiImageTextComboNode.INPUT_TYPES()
    assert "image_delivery" not in fields["optional"]
    assert list(fields["optional"])[-1] == "model"
    models, options = fields["optional"]["model"]
    assert models == ["qwen3.7-plus", "doubao-seed-2.0-mini"]
    assert options["default"] == "qwen3.7-plus"
    assert "FD_IMAGE_TEXT_PROVIDER" in options["tooltip"]


def test_qwen_payload_does_not_change_subsequent_default_request(monkeypatch):
    node = module.ZhiYiImageTextComboNode()
    messages = [{"role": "user", "content": []}]
    monkeypatch.setattr(module.config, "FD_IMAGE_TEXT_PROVIDER", "doubao")
    monkeypatch.setattr(module.config, "FD_IMAGE_TEXT_MODEL", "doubao-seed-2.0-mini")
    doubao_payload = node._build_request_payload(messages, 0.7, 100)
    assert doubao_payload["model"] == "doubao-seed-2.0-mini"
    assert "enable_thinking" not in doubao_payload
    monkeypatch.setattr(module.config, "FD_IMAGE_TEXT_PROVIDER", "qwen")
    monkeypatch.setattr(module.config, "FD_IMAGE_TEXT_MODEL", "qwen3.7-plus")
    assert node._build_request_payload(messages, 0.7, 100)["model"] == "qwen3.7-plus"
