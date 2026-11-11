"""Three public nodes use the dedicated provider, regardless of legacy model/address settings."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import requests
import torch

from src.Comfyui_Fd_Nodes import config, old_fd_nodes
from src.Comfyui_Fd_Nodes import zhiyi_image_text_combo_node as combo
from src.Comfyui_Fd_Nodes import zhiyi_image_text_node as single


@pytest.mark.parametrize("provider, expected", [(None, "qwen3.7-plus"), ("qwen", "qwen3.7-plus"),
                                               ("doubao", "doubao-seed-2.0-mini")])
@pytest.mark.parametrize("runtime", ["dev", "domestic", "overseas"])
def test_provider_startup_is_independent(provider: str | None, expected: str, runtime: str) -> None:
    env = dict(os.environ, FD_LITELLM_BASE_URL="https://chat.example.com", FD_LITELLM_API_KEY="test",
               FD_COMFY_RUNTIME_ENV=runtime, FD_GPT_IMAGE_BACKEND="image_generation", FD_IMAGE_DELIVERY="url",
               FD_DOUBAO_VISION_URL="https://doubao.example.com/vision")
    env.pop("FD_IMAGE_TEXT_PROVIDER", None)
    if provider is not None:
        env["FD_IMAGE_TEXT_PROVIDER"] = provider
    path = Path(config.__file__)
    result = subprocess.run(
        [sys.executable, "-c", "import runpy,sys; print(runpy.run_path(sys.argv[1])['FD_IMAGE_TEXT_MODEL'])", str(path)],
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("provider", ["qwen", "doubao"])
@pytest.mark.parametrize("node_kind", ["legacy", "single", "combo"])
@pytest.mark.parametrize("legacy_model", ["qwen3.7-plus", "doubao-seed-2.0-mini"])
def test_all_nodes_obey_provider(
    monkeypatch: pytest.MonkeyPatch, provider: str, node_kind: str, legacy_model: str,
) -> None:
    model = "qwen3.7-plus" if provider == "qwen" else "doubao-seed-2.0-mini"
    monkeypatch.setattr(config, "FD_IMAGE_TEXT_PROVIDER", provider)
    monkeypatch.setattr(config, "FD_IMAGE_TEXT_MODEL", model)
    monkeypatch.setattr(old_fd_nodes, "FD_DOUBAO_URL", "https://doubao.example.com/legacy")
    for module in (single, combo):
        monkeypatch.setattr(module, "FD_DOUBAO_VISION_URL", "https://doubao.example.com/vision")
    for module in (single, combo, old_fd_nodes):
        monkeypatch.setattr(module, "load_config", lambda: {"base_url": "https://chat.example.com", "api_key": "test"})
    monkeypatch.delenv("FD_OVERSEAS_GPU_GATEWAY", raising=False)
    monkeypatch.delenv("FD_IMAGE_DELIVERY", raising=False)
    sent = []

    class Response:
        ok = True
        status_code = 200
        text = "ok"
        content = b'{"status": true, "result_text": "description"}'

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"status": True, "result_text": "description",
                    "choices": [{"message": {"content": [{"type": "text", "text": "description"}]}}]}

    def post(*args, **kwargs) -> Response:
        sent.append((args[0] if args else kwargs["url"], kwargs.get("json") or json.loads(kwargs["data"])))
        return Response()

    monkeypatch.setattr(requests, "post", post)
    image = torch.zeros((1, 16, 16, 3))
    if node_kind == "legacy":
        output = old_fd_nodes.FD_imgToText_Doubao().gen("https://cdn.example.com/a.png", "describe", "fallback")
    elif node_kind == "single":
        output = single.ZhiYiImageTextNode().generate(image, "describe", node_switch=0)
    else:
        output = combo.ZhiYiImageTextComboNode().generate(
            combo_1={"images": [image], "prompt": "describe"}, retry_count=0, model=legacy_model,
        )
    assert output[0] == "description"
    assert len(sent) == 1
    url, body = sent[0]
    if provider == "qwen":
        assert url == "https://chat.example.com/v1/chat/completions"
        assert body["model"] == model and body["enable_thinking"] is False
        assert body["messages"][-1]["content"][0]["text"] == "describe"
        if node_kind == "legacy":
            assert body["messages"][-1]["content"][1]["image_url"]["url"] == "https://cdn.example.com/a.png"
    else:
        assert url == f"https://doubao.example.com/{'legacy' if node_kind == 'legacy' else 'vision'}"
        assert body["text_prompt"] == "describe"


def test_qwen_legacy_failure_keeps_default_without_doubao_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []

    def failed_post(*args, **kwargs) -> None:
        sent.append(args[0])
        raise requests.Timeout("upstream timeout")

    monkeypatch.setattr(old_fd_nodes, "load_config", lambda: {"base_url": "https://chat.example.com", "api_key": "test"})
    monkeypatch.setattr(requests, "post", failed_post)
    assert old_fd_nodes.FD_imgToText_Doubao().gen("https://cdn.example.com/a.png", "describe", "fallback") == ("fallback",)
    assert sent == ["https://chat.example.com/v1/chat/completions"]
