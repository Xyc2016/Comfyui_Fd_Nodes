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
from src.Comfyui_Fd_Nodes import zhiyi_image_text_combo_node as zhiyi_image_text_combo_module
from src.Comfyui_Fd_Nodes import zhiyi_image_text_node as zhiyi_image_text_module
from src.Comfyui_Fd_Nodes import zhiyi_text_node as zhiyi_text_module
from src.Comfyui_Fd_Nodes.old_gemini_api_node import FD_GeminiImage
from src.Comfyui_Fd_Nodes.zhiyi_image_text_combo_node import ZhiYiImageTextComboNode
from src.Comfyui_Fd_Nodes.zhiyi_image_text_node import ZhiYiImageTextNode
from src.Comfyui_Fd_Nodes.zhiyi_image_to_image_combo_node import ZhiYiImageToImageComboNode
from src.Comfyui_Fd_Nodes.zhiyi_image_to_image_node import ZhiYiImageToImageNode
from src.Comfyui_Fd_Nodes.zhiyi_text_node import ZhiYiTextGenNode
from src.Comfyui_Fd_Nodes.utils.error_utils import normalize_error_message
from src.Comfyui_Fd_Nodes.utils.gemini_service import (
    GeminiImageServiceClient,
    compose_prompt,
    normalize_gemini_model_name,
)
from src.Comfyui_Fd_Nodes.utils.litellm_gemini_image import should_use_litellm_gemini
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
    assert {"base_url", "api_key"}.isdisjoint(ZhiYiImageTextComboNode.INPUT_TYPES()["required"])
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


def test_zhiyi_image_to_image_nodes_expose_legacy_and_channel_models():
    expected_models = {
        "google/gemini-2.5-flash-image-preview",
        "google/gemini-3-pro-image-preview",
        "google/gemini-3-pro-image-preview-official",
        "google/gemini-3.1-flash-image-preview",
        "batch/gemini-3-pro-image-preview",
        "gemini-3-pro-image-preview-aistudio",
        "gemini-3-pro-image-preview-siphonlab",
    }

    assert expected_models.issubset(set(ZhiYiImageToImageNode.MODELS))
    assert expected_models.issubset(set(ZhiYiImageToImageComboNode.MODELS))
    assert "batch/gemini-3-pro-image-preview" in FD_GeminiImage.INPUT_TYPES()["required"]["model"][1]["options"]
    assert ZhiYiImageToImageNode.INPUT_TYPES()["required"]["model"][1]["default"] == "google/gemini-3-pro-image-preview"
    assert ZhiYiImageToImageComboNode.INPUT_TYPES()["required"]["model"][1]["default"] == "google/gemini-3-pro-image-preview"


def test_gemini_service_builds_internal_request_body():
    client = GeminiImageServiceClient(service_url="https://gemini.internal", oss_url_prefix="https://oss/")

    body = client.build_request_body(
        prompt="draw product",
        model="gemini-3-pro-image-preview",
        image_url_list=["https://oss/input.png"],
        aspect_ratio="1:1",
        image_size="4K",
        out_request_id="req-123",
    )

    assert body == {
        "out_request_id": "req-123",
        "prompt": "draw product",
        "model": "google/gemini-3-pro-image-preview",
        "aspect_ratio": "1:1",
        "image_url_list": ["https://oss/input.png"],
        "resolution": "4K",
    }
    assert client.summarize_request_body(body)["image_count"] == 1

    old_batch_body = client.build_request_body(
        prompt="draw product",
        model="batch/gemini-3-pro-image-preview",
        image_url_list=["https://oss/input.png"],
    )
    assert old_batch_body["model"] == "batch/gemini-3-pro-image-preview"

    aistudio_body = client.build_request_body(
        prompt="draw product",
        model="gemini-3-pro-image-preview-aistudio",
        image_url_list=["https://oss/input.png"],
    )
    assert aistudio_body["model"] == "google/gemini-3-pro-image-preview-official"


def test_gemini_service_model_and_prompt_helpers():
    assert normalize_gemini_model_name("gemini-2.5-flash-image-preview") == "google/gemini-2.5-flash-image-preview"
    assert normalize_gemini_model_name("google/gemini-3-pro-image-preview") == "google/gemini-3-pro-image-preview"
    assert normalize_gemini_model_name("batch/gemini-3-pro-image-preview") == "batch/gemini-3-pro-image-preview"
    assert normalize_gemini_model_name("gemini-3-pro-image-preview-official") == "google/gemini-3-pro-image-preview-official"
    assert normalize_gemini_model_name("google/gemini-3-pro-image-preview-official") == "google/gemini-3-pro-image-preview-official"
    assert normalize_gemini_model_name("gemini-3-pro-image-preview-aistudio") == "google/gemini-3-pro-image-preview-official"
    assert should_use_litellm_gemini("gemini-3-pro-image-preview-aistudio") is False
    assert should_use_litellm_gemini("batch/gemini-3-pro-image-preview") is False
    assert should_use_litellm_gemini("gemini-3-pro-image-preview-siphonlab") is True
    assert should_use_litellm_gemini("gemini-3-pro-image-preview") is False
    assert compose_prompt("user prompt", "system prompt") == "system prompt\n\nuser prompt"
    assert compose_prompt("user prompt", "") == "user prompt"


def test_zhiyi_image_to_image_uses_internal_gemini_service(monkeypatch):
    node = ZhiYiImageToImageNode()
    calls = []

    class FakeClient:
        def upload_images(self, image_tensors):
            calls.append(("upload_images", len(image_tensors)))
            return ["https://oss/input.png"]

        def call_with_image_urls(self, **kwargs):
            calls.append(("call", kwargs))
            return torch.ones((1, 2, 2, 3), dtype=torch.float32), "https://oss/result.png", "ok"

    node.gemini_client = FakeClient()

    result, actual_seed = node.generate(
        image_1=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        prompt="user prompt",
        model="gemini-3-pro-image-preview",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        seed_mode="固定种子",
        seed=42,
        out_request_id="req-123",
        system_prompt="system prompt",
    )

    assert result.shape == (1, 2, 2, 3)
    assert actual_seed == 42
    assert calls[0] == ("upload_images", 1)
    request = calls[1][1]
    assert request["image_url_list"] == ["https://oss/input.png"]
    assert request["prompt"] == "system prompt\n\nuser prompt"
    assert request["model"] == "gemini-3-pro-image-preview"
    assert request["aspect_ratio"] == "1:1"
    assert request["image_size"] == "4K"
    assert request["out_request_id"] == "req-123"


def test_zhiyi_image_to_image_aistudio_uses_internal_official_model(monkeypatch):
    node = ZhiYiImageToImageNode()
    calls = []

    class FakeClient:
        def upload_images(self, image_tensors):
            calls.append(("upload_images", len(image_tensors)))
            return ["https://oss/input.png"]

        def call_with_image_urls(self, **kwargs):
            calls.append(("call", kwargs))
            return torch.ones((1, 2, 2, 3), dtype=torch.float32), "https://oss/result.png", "ok"

    node.gemini_client = FakeClient()
    monkeypatch.setattr(node, "_single_litellm_request", lambda *args: pytest.fail("aistudio should use internal Gemini service"))

    result, actual_seed = node.generate(
        image_1=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        prompt="user prompt",
        model="gemini-3-pro-image-preview-aistudio",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        seed_mode="固定种子",
        seed=42,
        out_request_id="req-123",
        system_prompt="system prompt",
    )

    assert result.shape == (1, 2, 2, 3)
    assert actual_seed == 42
    assert calls[0] == ("upload_images", 1)
    request = calls[1][1]
    assert request["image_url_list"] == ["https://oss/input.png"]
    assert request["prompt"] == "system prompt\n\nuser prompt"
    assert request["model"] == "gemini-3-pro-image-preview-aistudio"
    assert normalize_gemini_model_name(request["model"]) == "google/gemini-3-pro-image-preview-official"


def test_normalize_error_message_classifies_timeout_and_nsfw():
    assert normalize_error_message("Read timed out") == "TIMEOUT: Read timed out"
    assert normalize_error_message("内容被过滤 (content_filter)，请修改提示词") == "NSFW: 内容被过滤 (content_filter)，请修改提示词"
    assert normalize_error_message("HTTP 400 from GPT Image API: bad request") == "UNKNOWN: HTTP 400 from GPT Image API: bad request"


def test_zhiyi_image_to_image_generate_returns_last_actual_error(monkeypatch):
    node = ZhiYiImageToImageNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _image_tensors: ["https://oss/input.png"])
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
    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _image_tensors: ["https://oss/input.png"])

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: API 请求失败: 500"):
        node.generate(
            model=ZhiYiImageToImageComboNode.MODELS[0],
            aspect_ratio="1:1",
            image_size="4K",
            batch_size=1,
            max_concurrency=1,
            combo_1=combo,
        )


def test_zhiyi_image_to_image_combo_uses_internal_gemini_service(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    node.gemini_client.service_url = "https://gemini.internal/generate"
    combo = {
        "images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "prompts": ["prompt one", "prompt two"],
    }
    calls = []

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda image_tensors: ["https://oss/input.png"])

    def fake_single_request(*args):
        calls.append(args)
        return torch.ones((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(node, "_single_request", fake_single_request)

    images, actual_seed, log_text = node.generate(
        model=ZhiYiImageToImageComboNode.MODELS[0],
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="req-456",
        combo_1=combo,
        system_prompt="system prompt",
    )

    assert len(images) == 2
    assert actual_seed == 7
    assert "url=https://gemini.internal/generate" in log_text
    assert calls[0] == (
        ["https://oss/input.png"],
        "system prompt\n\nprompt one",
        ZhiYiImageToImageComboNode.MODELS[0],
        "1:1",
        "4K",
        7,
        "req-456",
    )
    assert calls[1][1] == "system prompt\n\nprompt two"


def test_zhiyi_image_to_image_combo_aistudio_uses_internal_official_model(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    node.gemini_client.service_url = "https://gemini.internal/generate"
    combo = {
        "images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "prompts": ["prompt one"],
    }
    calls = []

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda image_tensors: calls.append(("upload_images", len(image_tensors))) or ["https://oss/input.png"])
    monkeypatch.setattr(node, "_single_litellm_request", lambda *args: pytest.fail("aistudio should use internal Gemini service"))

    def fake_single_request(*args):
        calls.append(("single_request", args))
        return torch.ones((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(node, "_single_request", fake_single_request)

    images, actual_seed, log_text = node.generate(
        model="gemini-3-pro-image-preview-aistudio",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="req-456",
        combo_1=combo,
        system_prompt="system prompt",
    )

    assert len(images) == 1
    assert actual_seed == 7
    assert "url=https://gemini.internal/generate" in log_text
    assert calls[0] == ("upload_images", 1)
    assert calls[1][0] == "single_request"
    request_args = calls[1][1]
    assert request_args == (
        ["https://oss/input.png"],
        "system prompt\n\nprompt one",
        "gemini-3-pro-image-preview-aistudio",
        "1:1",
        "4K",
        7,
        "req-456",
    )
    assert normalize_gemini_model_name(request_args[2]) == "google/gemini-3-pro-image-preview-official"


def test_zhiyi_image_to_image_combo_siphonlab_uses_litellm_without_model_rename(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    combo = {
        "images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "prompts": ["prompt one"],
    }
    calls = []

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.zhiyi_image_to_image_combo_node.tensor_to_base64", lambda _tensor: "AAA")
    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _image_tensors: pytest.fail("siphonlab should not upload images to Gemini service"))

    def fake_litellm(*args):
        calls.append(args)
        return torch.ones((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(node, "_single_litellm_request", fake_litellm)

    images, actual_seed, log_text = node.generate(
        model="gemini-3-pro-image-preview-siphonlab",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="req-456",
        combo_1=combo,
        system_prompt="system prompt",
    )

    assert len(images) == 1
    assert actual_seed == 7
    assert "FD_LITELLM_BASE_URL/v1/chat/completions" in log_text
    assert calls[0][1] == "gemini-3-pro-image-preview-siphonlab"
    assert calls[0][0][0]["role"] == "system"
    assert calls[0][0][1]["content"][0]["text"] == "prompt one"
    assert calls[0][0][1]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"


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

    def fake_image_tensor_to_data_url(_image):
        return (
            "data:image/jpeg;base64,AAA",
            {
                "original_size": (1, 1),
                "final_size": (1, 1),
                "mime_type": "image/jpeg",
                "quality": 85,
                "image_bytes": 3,
                "data_url_bytes": 26,
                "max_data_url_bytes": zhiyi_image_text_module.MAX_IMAGE_DATA_URL_BYTES,
                "resized": False,
            },
        )

    def fake_logger_info(message, *args):
        captured_logs.append((message, args))

    monkeypatch.setattr(zhiyi_image_text_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(zhiyi_image_text_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_image_tensor_to_data_url", fake_image_tensor_to_data_url)
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
    assert request_payload["model"] == "doubao-seed-2.0-mini"
    assert request_payload["messages"][1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
    request_log = next(args[0] for message, args in captured_logs if message == "Calling ZhiYi image-to-text API with payload=%s")
    assert request_log["messages"] == [
        {"role": "system", "text": "system prompt"},
        {"role": "user", "text": "describe this image", "image_count": 1},
    ]
    encoded_image_log = next(args[0] for message, args in captured_logs if message == "ZhiYi image-to-text encoded image: %s")
    assert encoded_image_log["mime_type"] == "image/jpeg"
    assert "data:image/jpeg;base64,AAA" not in json.dumps(request_log, ensure_ascii=False)
    response_log = next(args[0] for message, args in captured_logs if message == "ZhiYi image-to-text API response summary: %s")
    assert response_log["status_code"] == 200
    assert response_log["choice_count"] == 1


def test_zhiyi_image_text_encodes_small_image_as_jpeg_data_url():
    """Small images should be encoded as JPEG data URLs without resizing."""
    node = ZhiYiImageTextNode()
    image = torch.zeros((1, 16, 16, 3), dtype=torch.float32)

    data_url, info = node._image_tensor_to_data_url(image)

    assert data_url.startswith("data:image/jpeg;base64,")
    assert len(data_url.encode("utf-8")) <= zhiyi_image_text_module.MAX_IMAGE_DATA_URL_BYTES
    assert info["original_size"] == (16, 16)
    assert info["final_size"] == (16, 16)
    assert info["quality"] == 85
    assert info["resized"] is False


def test_zhiyi_image_text_recompresses_until_data_url_is_under_limit(monkeypatch):
    """Large/noisy inputs should be retried with lower quality and smaller dimensions until under the transport limit."""
    node = ZhiYiImageTextNode()
    image = torch.rand((1, 512, 512, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_module, "MAX_IMAGE_DATA_URL_BYTES", 50_000)
    monkeypatch.setattr(zhiyi_image_text_module, "MIN_IMAGE_LONG_EDGE", 128)
    monkeypatch.setattr(zhiyi_image_text_module, "IMAGE_RESIZE_FACTOR", 0.5)

    data_url, info = node._image_tensor_to_data_url(image)

    assert data_url.startswith("data:image/jpeg;base64,")
    assert len(data_url.encode("utf-8")) <= 50_000
    assert info["data_url_bytes"] <= 50_000
    assert info["final_size"][0] <= 512
    assert info["final_size"][1] <= 512
    assert info["resized"] is True


def test_zhiyi_image_text_raises_clear_error_when_image_cannot_fit(monkeypatch):
    """If every compression attempt remains too large, the node should fail before making the API request."""
    node = ZhiYiImageTextNode()
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_module, "MAX_IMAGE_DATA_URL_BYTES", 1)
    monkeypatch.setattr(zhiyi_image_text_module, "MIN_IMAGE_LONG_EDGE", 1)

    with pytest.raises(RuntimeError, match="10MB 传输限制"):
        node._image_tensor_to_data_url(image)


def test_zhiyi_image_text_combo_registered_and_accepts_combo_inputs():
    input_types = ZhiYiImageTextComboNode.INPUT_TYPES()

    assert "ZhiYiImageTextComboNode" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["ZhiYiImageTextComboNode"] is ZhiYiImageTextComboNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiImageTextComboNode"] == "知衣-图生文-combo"
    assert set(input_types["required"]) == {"max_concurrency", "temperature", "max_tokens", "retry_count"}
    assert "combo_1" in input_types["optional"]
    assert "combo_8" in input_types["optional"]
    assert ZhiYiImageTextComboNode.RETURN_TYPES == ("STRING", "STRING")
    assert ZhiYiImageTextComboNode.RETURN_NAMES == ("text", "log")


def test_zhiyi_image_text_combo_sends_multiple_images_in_one_request(monkeypatch):
    node = ZhiYiImageTextComboNode()
    captured_payloads = []
    captured_logs = []
    image = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
    combo = {"images": [image], "prompts": ["describe both images"]}

    class DummyResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"both images"}}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "both images"}}]}

    def fake_post(url, headers, data, timeout):
        captured_payloads.append(json.loads(data))
        return DummyResponse()

    def fake_image_tensor_to_data_url(image_tensor):
        index = len(captured_logs) + 1
        return (
            f"data:image/jpeg;base64,IMG{index}",
            {
                "original_size": (2, 2),
                "final_size": (2, 2),
                "mime_type": "image/jpeg",
                "quality": 85,
                "image_bytes": index,
                "data_url_bytes": 30,
                "max_data_url_bytes": zhiyi_image_text_module.MAX_IMAGE_DATA_URL_BYTES,
                "resized": False,
            },
        )

    def fake_logger_info(message, *args):
        captured_logs.append((message, args))

    monkeypatch.setattr(zhiyi_image_text_combo_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(zhiyi_image_text_combo_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_image_tensor_to_data_url", fake_image_tensor_to_data_url)
    monkeypatch.setattr(zhiyi_image_text_combo_module.logger, "info", fake_logger_info)

    text, log_text = node.generate(
        max_concurrency=1,
        temperature=0.2,
        max_tokens=256,
        retry_count=0,
        combo_1=combo,
        system_prompt="system prompt",
    )

    assert text == "both images"
    assert "总计: 1/1 成功" in log_text
    payload = captured_payloads[0]
    assert payload["model"] == "doubao-seed-2.0-mini"
    assert payload["messages"][0]["role"] == "system"
    content = payload["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "describe both images"}
    assert [part["image_url"]["url"] for part in content[1:]] == [
        "data:image/jpeg;base64,IMG1",
        "data:image/jpeg;base64,IMG2",
    ]
    request_log = next(args[0] for message, args in captured_logs if message == "Calling ZhiYi image-to-text combo API with payload=%s")
    assert request_log["messages"] == [
        {"role": "system", "text": "system prompt"},
        {"role": "user", "text": "describe both images", "image_count": 2},
    ]
    assert "data:image/jpeg;base64,IMG1" not in json.dumps(request_log, ensure_ascii=False)


def test_zhiyi_image_text_combo_keeps_output_order_when_requests_finish_out_of_order(monkeypatch):
    node = ZhiYiImageTextComboNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_combo_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(node, "_image_tensor_to_data_url", lambda _image: ("data:image/jpeg;base64,AAA", {}))

    def fake_single_request(url, api_key, messages, temperature, max_tokens, retry_count):
        prompt = messages[-1]["content"][0]["text"]
        if prompt == "first":
            return "text first", 0
        return "text second", 0

    def fake_run_concurrent(tasks, max_workers, label="请求"):
        second = tasks[1]
        first = tasks[0]
        results = {
            second[0]: {"ok": True, "text": second[2](*second[3])[0]},
            first[0]: {"ok": True, "text": first[2](*first[3])[0]},
        }
        return results, ["[请求 combo_2] 成功", "[请求 combo_1] 成功"], None

    monkeypatch.setattr(node, "_single_request", fake_single_request)
    monkeypatch.setattr(node, "_run_concurrent", fake_run_concurrent)

    text, log_text = node.generate(
        max_concurrency=2,
        retry_count=0,
        combo_1={"images": [image], "prompts": ["first"]},
        combo_2={"images": [image], "prompts": ["second"]},
    )

    assert text == "[combo_1]\ntext first\n\n[combo_2]\ntext second"
    assert "总计: 2/2 成功" in log_text


def test_zhiyi_image_text_combo_retries_once_after_retryable_failure(monkeypatch):
    node = ZhiYiImageTextComboNode()
    calls = []

    class DummyResponse:
        def __init__(self, status_code, text, payload=None):
            self.status_code = status_code
            self.text = text
            self.payload = payload or {"choices": [{"message": {"content": "retry success"}}]}

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_post(url, headers, data, timeout):
        calls.append(json.loads(data))
        if len(calls) == 1:
            return DummyResponse(500, "internal error")
        return DummyResponse(200, '{"choices":[{"message":{"content":"retry success"}}]}')

    monkeypatch.setattr(zhiyi_image_text_combo_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_sleep_before_retry", lambda _attempt_index: None)

    text, retry_used = node._single_request(
        url="https://example.com/v1/chat/completions",
        api_key="secret",
        messages=[{"role": "user", "content": [{"type": "text", "text": "prompt"}]}],
        temperature=0.7,
        max_tokens=128,
        retry_count=1,
    )

    assert text == "retry success"
    assert retry_used == 1
    assert len(calls) == 2


def test_zhiyi_image_text_combo_keeps_failure_placeholder_for_partial_failure(monkeypatch):
    node = ZhiYiImageTextComboNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_combo_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(node, "_image_tensor_to_data_url", lambda _image: ("data:image/jpeg;base64,AAA", {}))

    def fake_run_concurrent(tasks, max_workers, label="请求"):
        return (
            {
                0: {"ok": True, "text": "ok text"},
                1: {"ok": False, "text": "ERROR: TIMEOUT: request timed out"},
            },
            ["[请求 combo_1] 成功", "[请求 combo_2] 失败: TIMEOUT: request timed out"],
            "TIMEOUT: request timed out",
        )

    monkeypatch.setattr(node, "_run_concurrent", fake_run_concurrent)

    text, log_text = node.generate(
        max_concurrency=2,
        retry_count=0,
        combo_1={"images": [image], "prompts": ["first"]},
        combo_2={"images": [image], "prompts": ["second"]},
    )

    assert text == "[combo_1]\nok text\n\n[combo_2]\nERROR: TIMEOUT: request timed out"
    assert "总计: 1/2 成功" in log_text
    assert "[请求 combo_2] 失败: TIMEOUT: request timed out" in log_text


def test_zhiyi_image_text_combo_raises_when_all_requests_fail(monkeypatch):
    node = ZhiYiImageTextComboNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_combo_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(node, "_image_tensor_to_data_url", lambda _image: ("data:image/jpeg;base64,AAA", {}))
    monkeypatch.setattr(
        node,
        "_run_concurrent",
        lambda tasks, max_workers, label="请求": (
            {0: {"ok": False, "text": "ERROR: UNKNOWN: API 请求失败: 500"}},
            ["[请求 combo_1] 失败: UNKNOWN: API 请求失败: 500"],
            "UNKNOWN: API 请求失败: 500",
        ),
    )

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: API 请求失败: 500$"):
        node.generate(
            max_concurrency=1,
            retry_count=0,
            combo_1={"images": [image], "prompts": ["first"]},
        )


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

        def addHandler(self, handler):
            self.handlers.append(handler)

        def removeHandler(self, handler):
            if handler in self.handlers:
                self.handlers.remove(handler)

        def setLevel(self, _level):
            return None

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

        def addHandler(self, handler):
            self.handlers.append(handler)

        def removeHandler(self, handler):
            if handler in self.handlers:
                self.handlers.remove(handler)

        def setLevel(self, _level):
            return None

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
