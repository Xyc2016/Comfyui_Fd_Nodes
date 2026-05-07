#!/usr/bin/env python

"""Tests for `Comfyui_Fd_Nodes` package."""

import json
import logging
from io import BytesIO

import pytest
from src.Comfyui_Fd_Nodes import gpt_image_node as gpt_image_module
from src.Comfyui_Fd_Nodes.gpt_image_edit_node import GPTImageEditNode
from src.Comfyui_Fd_Nodes.gpt_image_node import FD_GTPImage
from src.Comfyui_Fd_Nodes.gpt_multi_image_node import FD_GPTMultiImage
from src.Comfyui_Fd_Nodes.nodes import (
    Example,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _resolution_to_edit_size,
)
from src.Comfyui_Fd_Nodes.old_gemini_api_node import GenImageServiceError
from src.Comfyui_Fd_Nodes.prompt_nodes import EcommercePromptGenerator, PromptListSelector
from src.Comfyui_Fd_Nodes import zhiyi_image_text_node as zhiyi_image_text_module
from src.Comfyui_Fd_Nodes import zhiyi_image_to_image_node as zhiyi_image_to_image_module
from src.Comfyui_Fd_Nodes import zhiyi_text_node as zhiyi_text_module
from src.Comfyui_Fd_Nodes.zhiyi_image_text_node import ZhiYiImageTextNode
from src.Comfyui_Fd_Nodes.zhiyi_image_to_image_node import ZhiYiImageToImageNode
from src.Comfyui_Fd_Nodes.zhiyi_text_node import ZhiYiTextGenNode
from src.Comfyui_Fd_Nodes.utils.logging_utils import (
    DEFAULT_LOG_DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    configure_default_logging,
)

@pytest.fixture
def example_node():
    """Fixture to create an Example node instance."""
    return Example()

def test_example_node_initialization(example_node):
    """Test that the node can be instantiated."""
    assert isinstance(example_node, Example)

def test_return_types():
    """Test the node's metadata."""
    assert Example.RETURN_TYPES == ("IMAGE",)
    assert Example.FUNCTION == "test"
    assert Example.CATEGORY == "Example"


def test_fd_gtp_image_metadata_and_size_mapping():
    """FD_GTPImage should stay registered and map UI presets to valid GPT sizes."""
    input_types = FD_GTPImage.INPUT_TYPES()

    assert "FD_GTPImage" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["FD_GTPImage"] is FD_GTPImage
    assert NODE_DISPLAY_NAME_MAPPINGS["FD_GTPImage"] == "FD GTP Image"

    assert set(input_types["required"]) == {"out_request_id", "prompt", "model", "resolution", "seed"}
    assert set(input_types["optional"]) == {"images", "files", "aspect_ratio"}
    assert FD_GTPImage.RETURN_TYPES == ("IMAGE", "STRING", "STRING")
    assert FD_GTPImage.FUNCTION == "api_call"
    assert FD_GTPImage.CATEGORY == "image/generation"

    assert _resolution_to_edit_size("1K", "") == "1024x1024"
    assert _resolution_to_edit_size("1K", "3:4") == "768x1024"
    assert _resolution_to_edit_size("1K", "4:3") == "1024x768"
    assert _resolution_to_edit_size("1K", "16:9") == "1280x720"
    assert _resolution_to_edit_size("1K", "9:16") == "720x1280"
    assert _resolution_to_edit_size("2K", "") == "2048x2048"
    assert _resolution_to_edit_size("2K", "3:4") == "1536x2048"
    assert _resolution_to_edit_size("2K", "4:3") == "2048x1536"
    assert _resolution_to_edit_size("2K", "16:9") == "2048x1152"
    assert _resolution_to_edit_size("2K", "9:16") == "1152x2048"
    assert _resolution_to_edit_size("4K", "") == "2880x2880"
    assert _resolution_to_edit_size("4K", "1:1") == "2880x2880"
    assert _resolution_to_edit_size("4K", "3:4") == "2160x2880"
    assert _resolution_to_edit_size("4K", "4:3") == "2880x2160"
    assert _resolution_to_edit_size("4K", "16:9") == "3840x2160"
    assert _resolution_to_edit_size("4K", "9:16") == "2160x3840"


def test_fd_gtp_image_requires_input_image():
    """FD_GTPImage should fail fast before making a network request when no image is provided."""
    node = FD_GTPImage()
    with pytest.raises(ValueError, match="requires at least one input image"):
        node.api_call(
            out_request_id="req-test",
            prompt="test prompt",
            model="gpt-image-2",
            resolution="2K",
            images=None,
            aspect_ratio="",
        )


class _FakeImageBatch:
    shape = (1, 2, 2, 3)

    def __getitem__(self, _slice):
        return self

    def squeeze(self):
        return self


def test_fd_gtp_image_timeout_falls_back_to_azure(monkeypatch):
    """Timeouts should skip further primary retries and immediately fall back to the Azure model."""
    node = FD_GTPImage()
    attempted_models = []
    fake_images = _FakeImageBatch()
    azure_calls = []

    def fake_request(*, data, multipart_files, batch_size):
        attempted_models.append(data["model"])
        if data["model"] == "gpt-image-2":
            raise GenImageServiceError("TIMEOUT")
        return BytesIO(b"fake-image"), "revised", "https://example.com/result.png"

    def fake_azure_request(*, data, multipart_files, batch_size):
        azure_calls.append({
            "data": data.copy(),
            "multipart_len": len(multipart_files),
            "batch_size": batch_size,
        })
        return BytesIO(b"fake-image"), "revised", "https://example.com/result.png"

    monkeypatch.setattr(node, "_request_gpt_image_edit", fake_request)
    monkeypatch.setattr(node, "_request_azure_gpt_image_generation", fake_azure_request)
    monkeypatch.setattr(gpt_image_module, "downscale_image_tensor", lambda image, total_pixels: image)
    monkeypatch.setattr(gpt_image_module, "_image_tensor_to_png_bytes", lambda _image: b"fake-png")
    monkeypatch.setattr(
        gpt_image_module,
        "bytesio_to_image_tensor",
        lambda _bytes: "fake-tensor",
    )

    result = node.api_call(
        out_request_id="req-timeout",
        prompt="test prompt",
        model="gpt-image-2",
        resolution="2K",
        images=fake_images,
        aspect_ratio="",
    )

    assert attempted_models == ["gpt-image-2"]
    assert azure_calls == [{
        "data": {
            "model": "gpt-image-2-azure",
            "prompt": "test prompt",
            "size": "2048x2048",
            "user": "req-timeout",
            "quality": "medium",
        },
        "multipart_len": 1,
        "batch_size": 1,
    }]
    assert result == ("fake-tensor", "revised", "https://example.com/result.png")


def test_fd_gtp_image_non_timeout_retries_primary_three_times_before_azure(monkeypatch):
    """Non-timeout failures should retry the primary model three times before using the Azure fallback."""
    node = FD_GTPImage()
    attempted_models = []
    fake_images = _FakeImageBatch()
    azure_calls = []

    def fake_request(*, data, multipart_files, batch_size):
        attempted_models.append(data["model"])
        if data["model"] == "gpt-image-2":
            raise GenImageServiceError("HTTP 500 from GPT Image API: boom")
        return BytesIO(b"fake-image"), "revised", "https://example.com/result.png"

    def fake_azure_request(*, data, multipart_files, batch_size):
        azure_calls.append({
            "data": data.copy(),
            "multipart_len": len(multipart_files),
            "batch_size": batch_size,
        })
        return BytesIO(b"fake-image"), "revised", "https://example.com/result.png"

    monkeypatch.setattr(node, "_request_gpt_image_edit", fake_request)
    monkeypatch.setattr(node, "_request_azure_gpt_image_generation", fake_azure_request)
    monkeypatch.setattr(gpt_image_module, "downscale_image_tensor", lambda image, total_pixels: image)
    monkeypatch.setattr(gpt_image_module, "_image_tensor_to_png_bytes", lambda _image: b"fake-png")
    monkeypatch.setattr(
        gpt_image_module,
        "bytesio_to_image_tensor",
        lambda _bytes: "fake-tensor",
    )

    result = node.api_call(
        out_request_id="req-http",
        prompt="test prompt",
        model="gpt-image-2",
        resolution="2K",
        images=fake_images,
        aspect_ratio="",
    )

    assert attempted_models == [
        "gpt-image-2",
        "gpt-image-2",
        "gpt-image-2",
    ]
    assert azure_calls == [{
        "data": {
            "model": "gpt-image-2-azure",
            "prompt": "test prompt",
            "size": "2048x2048",
            "user": "req-http",
            "quality": "medium",
        },
        "multipart_len": 1,
        "batch_size": 1,
    }]
    assert result == ("fake-tensor", "revised", "https://example.com/result.png")


def test_fd_gtp_multi_image_metadata():
    """FD_GTPMultiImage should be registered with ZhiYi-like inputs."""
    input_types = FD_GPTMultiImage.INPUT_TYPES()

    assert "FD_GTPMultiImage" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["FD_GTPMultiImage"] is FD_GPTMultiImage
    assert NODE_DISPLAY_NAME_MAPPINGS["FD_GTPMultiImage"] == "FD GTP Multi Image"

    required_inputs = input_types["required"]
    optional_inputs = input_types["optional"]

    assert set(required_inputs) == {
        "image_1",
        "prompt",
        "model",
        "aspect_ratio",
        "image_size",
        "batch_size",
        "seed_mode",
        "seed",
    }
    assert set(optional_inputs) == {
        "node_switch",
        "out_request_id",
        "prompt_list",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
        "image_6",
        "system_prompt",
    }
    assert FD_GPTMultiImage.RETURN_TYPES == ("IMAGE", "INT")
    assert FD_GPTMultiImage.FUNCTION == "generate"


def test_fd_gtp_multi_image_size_mapping_matches_shared_gpt_resolution_logic():
    """FD_GTPMultiImage should reuse the shared GPT size mapping."""
    node = FD_GPTMultiImage()

    assert node._build_gpt_size("1:1", "4K") == _resolution_to_edit_size("4K", "1:1")
    assert node._build_gpt_size("3:4", "2K") == _resolution_to_edit_size("2K", "3:4")
    assert node._build_gpt_size("4:3", "2K") == _resolution_to_edit_size("2K", "4:3")
    assert node._build_gpt_size("16:9", "4K") == _resolution_to_edit_size("4K", "16:9")
    assert node._build_gpt_size("9:16", "4K") == _resolution_to_edit_size("4K", "9:16")
    assert node._build_gpt_size("1:1", "720P") == _resolution_to_edit_size("1K", "1:1")
    assert node._build_gpt_size("16:9", "1080P") == _resolution_to_edit_size("1K", "16:9")


def test_fd_gtp_multi_image_exposes_out_request_id():
    """FD_GTPMultiImage should expose out_request_id like ZhiYiImageToImageNode."""
    optional_inputs = FD_GPTMultiImage.INPUT_TYPES()["optional"]

    assert "out_request_id" in optional_inputs
    assert optional_inputs["out_request_id"][1]["default"] == "default"


def test_fd_gtp_multi_image_generate_expands_prompt_list_and_batch(monkeypatch):
    """FD_GTPMultiImage should expand prompt_list x batch_size using ZhiYi-style scheduling."""
    node = FD_GPTMultiImage()
    torch = pytest.importorskip("torch")
    dummy_image = torch.ones((1, 8, 8, 3), dtype=torch.float32)
    calls = []

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_multi_image_node.load_config",
        lambda: {"base_url": "https://example.com", "api_key": "secret"},
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_multi_image_node.FD_LITELLM_BASE_URL",
        "https://override.example.com",
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_multi_image_node.FD_LITELLM_API_KEY",
        "override-secret",
    )

    def fake_single_request(url, api_key, model, prompt, images, aspect_ratio, image_size, out_request_id):
        calls.append(
            {
                "url": url,
                "api_key": api_key,
                "model": model,
                "prompt": prompt,
                "images_len": len(images),
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "out_request_id": out_request_id,
            }
        )
        value = len(calls) / 10.0
        return torch.full((1, 4, 4, 4), value, dtype=torch.float32)

    monkeypatch.setattr(node, "_single_request", fake_single_request)
    monkeypatch.setattr(node, "_run_concurrent", lambda tasks, label="任务": [fn(*args) for _, fn, args in tasks])

    output, actual_seed = node.generate(
        image_1=dummy_image,
        image_2=dummy_image,
        prompt="fallback prompt",
        model="gpt-image-2",
        aspect_ratio="3:4",
        image_size="2K",
        batch_size=2,
        seed_mode="固定种子",
        seed=123,
        node_switch=0,
        out_request_id="req-123",
        prompt_list=["p1", "p2"],
        system_prompt="system prompt",
    )

    assert actual_seed == 123
    assert output.shape[0] == 4
    assert len(calls) == 4
    assert all(call["url"] == "https://override.example.com/v1/images/edits" for call in calls)
    assert all(call["api_key"] == "override-secret" for call in calls)
    assert all(call["model"] == "gpt-image-2" for call in calls)
    assert all(call["images_len"] == 2 for call in calls)
    assert all(call["aspect_ratio"] == "3:4" for call in calls)
    assert all(call["image_size"] == "2K" for call in calls)
    assert all(call["out_request_id"] == "req-123" for call in calls)
    assert [call["prompt"] for call in calls] == [
        "system prompt\n\np1",
        "system prompt\n\np1",
        "system prompt\n\np2",
        "system prompt\n\np2",
    ]


def test_new_nodes_hide_base_url_and_api_key_inputs():
    """New custom nodes should read API config from project settings instead of exposing it in the UI."""
    assert {"base_url", "api_key"}.isdisjoint(GPTImageEditNode.INPUT_TYPES()["required"])
    assert {"base_url", "api_key"}.isdisjoint(FD_GPTMultiImage.INPUT_TYPES()["required"])
    assert {"api_url", "api_key"}.isdisjoint(EcommercePromptGenerator.INPUT_TYPES()["required"])
    assert {"base_url", "api_key"}.isdisjoint(ZhiYiImageTextNode.INPUT_TYPES()["required"])
    assert {"base_url", "api_key"}.isdisjoint(ZhiYiImageToImageNode.INPUT_TYPES()["required"])
    assert {"base_url", "api_key"}.isdisjoint(ZhiYiTextGenNode.INPUT_TYPES()["required"])


def test_selector_node_never_exposed_api_inputs():
    """PromptListSelector should remain a pure list-selection node."""
    assert {"base_url", "api_key", "api_url"}.isdisjoint(PromptListSelector.INPUT_TYPES()["required"])


def test_zhiyi_image_to_image_exposes_out_request_id():
    """ZhiYiImageToImageNode should expose out_request_id in the optional UI inputs."""
    optional_inputs = ZhiYiImageToImageNode.INPUT_TYPES()["optional"]

    assert "out_request_id" in optional_inputs
    assert optional_inputs["out_request_id"][1]["default"] == "default"


def test_zhiyi_image_to_image_request_logs_request_and_response_without_images(monkeypatch):
    """The image-to-image API should log request/response summaries without image payloads."""
    node = ZhiYiImageToImageNode()
    captured_logs = []

    class DummyResponse:
        ok = True
        status_code = 200
        text = '{"choices":[{"finish_reason":"stop","message":{"images":[{"image_url":{"url":"https://example.com/img.png"}}]}}]}'

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "images": [
                                {"image_url": {"url": "https://example.com/img.png"}}
                            ]
                        },
                    }
                ]
            }

    def fake_post(url, headers, data, timeout):
        captured_logs.append(("post_payload", json.loads(data)))
        return DummyResponse()

    def fake_extract(_result):
        return "data:image/png;base64,ZmFrZQ=="

    def fake_base64_to_tensor(data_url):
        captured_logs.append(("data_url", data_url))
        return "fake-tensor"

    def fake_logger_info(message, *args):
        captured_logs.append((message, args))

    monkeypatch.setattr(zhiyi_image_to_image_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_extract_image_from_response", fake_extract)
    monkeypatch.setattr(node, "_base64_to_tensor", fake_base64_to_tensor)
    monkeypatch.setattr(zhiyi_image_to_image_module.logger, "info", fake_logger_info)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "user prompt"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ],
        },
    ]

    result = node._single_request(
        url="https://example.com/v1/chat/completions",
        api_key="secret",
        out_request_id="req-123",
        messages=messages,
        model="custom-model",
        aspect_ratio="1:1",
        image_size="4K",
        seed=42,
    )

    assert result == "fake-tensor"
    request_payload = next(value for key, value in captured_logs if key == "post_payload")
    assert request_payload["user"] == "req-123"
    assert request_payload["seed"] == 42

    request_log = next(args[0] for message, args in captured_logs if message == "Calling ZhiYi image-to-image API with payload=%s")
    assert request_log["user"] == "req-123"
    assert request_log["messages"] == [
        {"role": "system", "text": "system prompt"},
        {"role": "user", "text": "user prompt", "image_count": 1},
    ]
    assert "data:image/png;base64,AAA" not in json.dumps(request_log, ensure_ascii=False)

    response_log = next(args[0] for message, args in captured_logs if message == "ZhiYi image-to-image API response summary: %s")
    assert response_log["status_code"] == 200
    assert response_log["choice_count"] == 1
    assert response_log["image_count"] == 1


def test_zhiyi_image_text_request_logs_request_and_response_without_images(monkeypatch):
    """The image-text API should log request/response summaries without image payloads."""
    node = ZhiYiImageTextNode()
    captured_logs = []

    class DummyResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"hello"}}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

    def fake_post(url, headers, data, timeout):
        captured_logs.append(("post_payload", json.loads(data)))
        return DummyResponse()

    def fake_image_tensor_to_base64(_image):
        return "AAA"

    def fake_logger_info(message, *args):
        captured_logs.append((message, args))

    monkeypatch.setattr(zhiyi_image_text_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(zhiyi_image_text_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_image_tensor_to_base64", fake_image_tensor_to_base64)
    monkeypatch.setattr(zhiyi_image_text_module.logger, "info", fake_logger_info)

    result = node.generate(
        image="fake-image",
        prompt="describe this image",
        node_switch=0,
        system_prompt="system prompt",
        temperature=0.2,
        max_tokens=256,
    )

    assert result == ("hello",)
    request_payload = next(value for key, value in captured_logs if key == "post_payload")
    assert request_payload["model"] == "gemini-3-pro-preview"
    request_log = next(args[0] for message, args in captured_logs if message == "Calling ZhiYi image-to-text API with payload=%s")
    assert request_log["messages"] == [
        {"role": "system", "text": "system prompt"},
        {"role": "user", "text": "describe this image", "image_count": 1},
    ]
    assert "data:image/png;base64,AAA" not in json.dumps(request_log, ensure_ascii=False)
    response_log = next(args[0] for message, args in captured_logs if message == "ZhiYi image-to-text API response summary: %s")
    assert response_log["status_code"] == 200
    assert response_log["choice_count"] == 1


def test_zhiyi_text_gen_request_logs_request_and_response(monkeypatch):
    """The text API should log request/response summaries."""
    node = ZhiYiTextGenNode()
    captured_logs = []

    class DummyResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"hello text"}}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hello text"}}]}

    def fake_post(url, headers, data, timeout):
        captured_logs.append(("post_payload", json.loads(data)))
        return DummyResponse()

    def fake_logger_info(message, *args):
        captured_logs.append((message, args))

    monkeypatch.setattr(zhiyi_text_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(zhiyi_text_module.requests, "post", fake_post)
    monkeypatch.setattr(zhiyi_text_module.logger, "info", fake_logger_info)

    result = node.generate(
        prompt="say hello",
        node_switch=0,
        system_prompt="system prompt",
        temperature=0.3,
        max_tokens=128,
    )

    assert result == ("hello text",)
    request_payload = next(value for key, value in captured_logs if key == "post_payload")
    assert request_payload["model"] == "gemini-3-pro-preview"
    request_log = next(args[0] for message, args in captured_logs if message == "Calling ZhiYi text API with payload=%s")
    assert request_log["messages"] == [
        {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
        {"role": "user", "content": [{"type": "text", "text": "say hello"}]},
    ]
    response_log = next(args[0] for message, args in captured_logs if message == "ZhiYi text API response summary: %s")
    assert response_log["status_code"] == 200
    assert response_log["choice_count"] == 1


def test_configure_default_logging_installs_timestamp_format(monkeypatch):
    """Default logging should include timestamps when no root handlers are configured."""
    captured = {}

    class FakeRootLogger:
        handlers = []

    def fake_get_logger(name=None):
        if name is None:
            return FakeRootLogger()
        raise AssertionError("Only the root logger should be requested")

    def fake_basic_config(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.utils.logging_utils.logging.getLogger", fake_get_logger)
    monkeypatch.setattr("src.Comfyui_Fd_Nodes.utils.logging_utils.logging.basicConfig", fake_basic_config)

    configure_default_logging()

    assert captured["level"] == 20
    assert captured["format"] == DEFAULT_LOG_FORMAT
    assert captured["datefmt"] == DEFAULT_LOG_DATE_FORMAT


def test_configure_default_logging_upgrades_existing_handler_without_timestamp(monkeypatch):
    """Existing root handlers without timestamps should be upgraded to the default formatter."""

    class FakeHandler:
        def __init__(self):
            self.formatter = logging.Formatter("%(message)s")

        def setFormatter(self, formatter):
            self.formatter = formatter

    class FakeRootLogger:
        def __init__(self):
            self.handlers = [FakeHandler()]

    fake_root_logger = FakeRootLogger()

    def fake_get_logger(name=None):
        if name is None:
            return fake_root_logger
        raise AssertionError("Only the root logger should be requested")

    basic_config_called = False

    def fake_basic_config(**kwargs):
        nonlocal basic_config_called
        basic_config_called = True

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.utils.logging_utils.logging.getLogger", fake_get_logger)
    monkeypatch.setattr("src.Comfyui_Fd_Nodes.utils.logging_utils.logging.basicConfig", fake_basic_config)

    configure_default_logging()

    assert basic_config_called is False
    assert fake_root_logger.handlers[0].formatter._fmt == DEFAULT_LOG_FORMAT
    assert fake_root_logger.handlers[0].formatter.datefmt == DEFAULT_LOG_DATE_FORMAT
