#!/usr/bin/env python

"""Tests for `Comfyui_Fd_Nodes` package."""

import json
import logging

import pytest
import torch
from src.Comfyui_Fd_Nodes.gpt_image_edit_node import GPTImageEditNode
from src.Comfyui_Fd_Nodes.gpt_multi_image_node import FD_GPTMultiImage
from src.Comfyui_Fd_Nodes.nodes import (
    FD_GTPImage,
    Example,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _resolution_to_edit_size,
)
from src.Comfyui_Fd_Nodes.prompt_nodes import EcommercePromptGenerator, PromptListSelector
from src.Comfyui_Fd_Nodes import zhiyi_image_text_node as zhiyi_image_text_module
from src.Comfyui_Fd_Nodes import zhiyi_image_to_image_node as zhiyi_image_to_image_module
from src.Comfyui_Fd_Nodes import zhiyi_text_node as zhiyi_text_module
from src.Comfyui_Fd_Nodes.zhiyi_image_text_node import ZhiYiImageTextNode
from src.Comfyui_Fd_Nodes.zhiyi_image_to_image_combo_node import ZhiYiImageToImageComboNode
from src.Comfyui_Fd_Nodes.zhiyi_image_to_image_node import ZhiYiImageToImageNode
from src.Comfyui_Fd_Nodes.zhiyi_text_node import ZhiYiTextGenNode
from src.Comfyui_Fd_Nodes.utils.error_utils import normalize_error_message
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
    assert _resolution_to_edit_size("1K", "9:16") == "720x1280"
    assert _resolution_to_edit_size("2K", "") == "2048x2048"
    assert _resolution_to_edit_size("2K", "3:4") == "1536x2048"
    assert _resolution_to_edit_size("2K", "9:16") == "1152x2048"
    assert _resolution_to_edit_size("4K", "") == "2880x2880"
    assert _resolution_to_edit_size("4K", "1:1") == "2880x2880"
    assert _resolution_to_edit_size("4K", "3:4") == "2160x2880"
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


def test_new_nodes_hide_base_url_and_api_key_inputs():
    """New custom nodes should read API config from project settings instead of exposing it in the UI."""
    assert {"base_url", "api_key"}.isdisjoint(GPTImageEditNode.INPUT_TYPES()["required"])
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


def test_normalize_error_message_classifies_timeout_and_nsfw():
    assert normalize_error_message("Read timed out") == "TIMEOUT: Read timed out"
    assert normalize_error_message("内容被过滤 (content_filter)，请修改提示词") == "NSFW: 内容被过滤 (content_filter)，请修改提示词"
    assert normalize_error_message("HTTP 400 from GPT Image API: bad request") == "UNKNOWN: HTTP 400 from GPT Image API: bad request"


def test_zhiyi_image_to_image_generate_returns_last_actual_error(monkeypatch):
    node = ZhiYiImageToImageNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_to_image_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(node, "_tensor_to_base64", lambda _tensor: "AAA")
    monkeypatch.setattr(
        node,
        "_run_concurrent",
        lambda tasks, label="任务": ([None] * len(tasks), "NSFW: 内容被过滤 (content_filter)，请修改提示词或输入图片后重试"),
    )

    with pytest.raises(RuntimeError, match=r"^NSFW: 内容被过滤"):
        node.generate(
            image_1=image,
            prompt="test prompt",
            model=ZhiYiImageToImageNode.MODELS[0],
            aspect_ratio="1:1",
            image_size="2K",
            batch_size=1,
        )


def test_gpt_multi_image_generate_returns_last_actual_error(monkeypatch):
    node = FD_GPTMultiImage()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.gpt_multi_image_node.load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(
        node,
        "_run_concurrent",
        lambda tasks, label="任务": ([None] * len(tasks), "TIMEOUT: request timed out"),
    )

    with pytest.raises(RuntimeError, match=r"^TIMEOUT: request timed out$"):
        node.generate(
            image_1=image,
            prompt="test prompt",
            model="gpt-image-2",
            aspect_ratio="1:1",
            image_size="2K",
            batch_size=1,
        )


def test_zhiyi_image_to_image_combo_generate_returns_last_actual_error(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    combo = {"images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)], "prompts": ["test prompt"]}

    monkeypatch.setattr(
        node,
        "_run_concurrent",
        lambda tasks, max_workers, label="任务": ([None] * len(tasks), ["[请求 1] 失败"], "UNKNOWN: API 请求失败: 500\ninternal error"),
    )
    monkeypatch.setattr(node, "_tensor_to_base64", lambda _tensor: "AAA")

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: API 请求失败: 500"):
        node.generate(
            model=ZhiYiImageToImageComboNode.MODELS[0],
            aspect_ratio="1:1",
            image_size="4K",
            batch_size=1,
            max_concurrency=1,
            combo_1=combo,
        )


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
