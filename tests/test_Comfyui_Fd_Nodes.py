#!/usr/bin/env python

"""Tests for `Comfyui_Fd_Nodes` package."""

import base64
import inspect
import io
import json
import logging

import pytest
import torch
from PIL import Image
from src.Comfyui_Fd_Nodes.gpt_image_edit_node import GPTImageEditNode
from src.Comfyui_Fd_Nodes.gpt_image_combo_node import FD_GPTImageComboNode
from src.Comfyui_Fd_Nodes.gpt_multi_image_node import FD_GPTMultiImage
from src.Comfyui_Fd_Nodes.nodes import (
    FD_GTPImage,
    Example,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _resolution_to_edit_size,
)
from src.Comfyui_Fd_Nodes.utils.gpt_image_size import resolution_to_edit_size
from src.Comfyui_Fd_Nodes.prompt_nodes import EcommercePromptGenerator, PromptListSelector
from src.Comfyui_Fd_Nodes import zhiyi_image_text_combo_node as zhiyi_image_text_combo_module
from src.Comfyui_Fd_Nodes import zhiyi_image_text_node as zhiyi_image_text_module
from src.Comfyui_Fd_Nodes import zhiyi_text_node as zhiyi_text_module
from src.Comfyui_Fd_Nodes.old_gemini_api_node import FD_GeminiImage, GeminiImageModel
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

    assert list(input_types["required"]) == ["out_request_id", "prompt", "model", "resolution", "seed"]
    assert list(input_types["optional"]) == ["images", "files", "aspect_ratio", "quality", "resize", "size_override"]
    assert FD_GTPImage.RETURN_TYPES == ("IMAGE", "STRING", "STRING")
    assert FD_GTPImage.FUNCTION == "api_call"
    assert FD_GTPImage.CATEGORY == "image/generation"

    assert _resolution_to_edit_size("1K", "") == "1K"
    assert _resolution_to_edit_size("1K", "3:4") == "1K"
    assert _resolution_to_edit_size("1K", "9:16") == "1K"
    assert _resolution_to_edit_size("2K", "") == "2K"
    assert _resolution_to_edit_size("2K", "3:4") == "2K"
    assert _resolution_to_edit_size("2K", "9:16") == "2K"
    assert _resolution_to_edit_size("4K", "") == "4K"
    assert _resolution_to_edit_size("4K", "1:1") == "4K"
    assert _resolution_to_edit_size("4K", "3:4") == "4K"
    assert _resolution_to_edit_size("4K", "9:16") == "2160x3840"


def test_gpt_aspect_ratios_include_9_21():
    """All GPT nodes and the size map should include the 9:21 aspect ratio."""
    # FD_GPTMultiImage
    assert "9:21" in FD_GPTMultiImage.ASPECT_RATIOS
    # FD_GPTImageComboNode
    assert "9:21" in FD_GPTImageComboNode.ASPECT_RATIOS
    # FD_GTPImage (uses IO.COMBO options)
    gtp_image_aspect_ratios = FD_GTPImage.INPUT_TYPES()["optional"]["aspect_ratio"][1]["options"]
    assert "9:21" in gtp_image_aspect_ratios

    # Size map values
    assert resolution_to_edit_size("1K", "9:21") == "624x1456"
    assert resolution_to_edit_size("2K", "9:21") == "864x2016"
    assert resolution_to_edit_size("4K", "9:21") == "1648x3840"


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


def test_gpt_nodes_keep_legacy_widget_order():
    """New optional controls must not shift saved workflow widget values."""
    single_inputs = FD_GTPImage.INPUT_TYPES()
    combo_inputs = FD_GPTImageComboNode.INPUT_TYPES()
    multi_inputs = FD_GPTMultiImage.INPUT_TYPES()

    assert list(single_inputs["required"]) == [
        "out_request_id",
        "prompt",
        "model",
        "resolution",
        "seed",
    ]
    assert list(single_inputs["optional"]) == ["images", "files", "aspect_ratio", "quality", "resize", "size_override"]

    assert list(combo_inputs["required"]) == [
        "model",
        "aspect_ratio",
        "image_size",
        "batch_size",
        "max_concurrency",
        "seed_mode",
        "seed",
    ]
    assert list(combo_inputs["optional"])[-2:] == ["resize", "size_override"]

    assert list(multi_inputs["required"]) == [
        "image_1",
        "prompt",
        "model",
        "aspect_ratio",
        "image_size",
        "batch_size",
        "seed_mode",
        "seed",
    ]
    assert list(multi_inputs["optional"])[-2:] == ["resize", "size_override"]


def test_fd_gtp_image_passes_resize_to_client(monkeypatch):
    node = FD_GTPImage()
    captured = {}

    class FakeClient:
        def edit_image(self, **kwargs):
            captured.update(kwargs)
            return io.BytesIO(b"fake-image"), "ok", "https://example.com/result.png"

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_node.get_default_gpt_image_edit_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_node.bytesio_to_image_tensor",
        lambda _image_bytesio: torch.ones((1, 2, 2, 3), dtype=torch.float32),
    )

    image, output_text, result_url = node.api_call(
        out_request_id="req-resize",
        prompt="edit image",
        model="gpt-image-2",
        resolution="2K",
        quality="high",
        resize=False,
        images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        aspect_ratio="3:4",
    )

    assert tuple(image.shape) == (1, 2, 2, 3)
    assert output_text == "ok"
    assert result_url == "https://example.com/result.png"
    assert captured["resize"] is False
    assert captured["aspect_ratio"] == "3:4"
    assert captured["quality"] == "high"
    assert captured["out_request_id"] == "req-resize"


def test_fd_gtp_image_size_override_passes_custom_size_to_client(monkeypatch):
    node = FD_GTPImage()
    captured = {}

    class FakeClient:
        def edit_image(self, **kwargs):
            captured.update(kwargs)
            return io.BytesIO(b"fake-image"), "ok", "https://example.com/result.png"

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_node.get_default_gpt_image_edit_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_node.bytesio_to_image_tensor",
        lambda _image_bytesio: torch.ones((1, 2, 2, 3), dtype=torch.float32),
    )

    node.api_call(
        out_request_id="req-override",
        prompt="edit",
        model="gpt-image-2",
        resolution="2K",
        images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        aspect_ratio="9:16",
        size_override="1537x1025",
    )

    assert captured["size"] == "1537x1025"
    assert captured["aspect_ratio"] == ""


def test_fd_gtp_image_no_override_keeps_preset_behavior(monkeypatch):
    node = FD_GTPImage()
    captured = {}

    class FakeClient:
        def edit_image(self, **kwargs):
            captured.update(kwargs)
            return io.BytesIO(b"fake-image"), "ok", "https://example.com/result.png"

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_node.get_default_gpt_image_edit_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_node.bytesio_to_image_tensor",
        lambda _image_bytesio: torch.ones((1, 2, 2, 3), dtype=torch.float32),
    )

    node.api_call(
        out_request_id="req-preset",
        prompt="edit",
        model="gpt-image-2",
        resolution="2K",
        images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        aspect_ratio="3:4",
    )

    assert captured["size"] == "2K"
    assert captured["aspect_ratio"] == "3:4"


def test_gpt_combo_single_request_passes_resize_to_client(monkeypatch):
    node = FD_GPTImageComboNode()
    captured = {}

    class FakeClient:
        def edit_image(self, **kwargs):
            captured.update(kwargs)
            return io.BytesIO(b"fake-image"), "ok", "https://example.com/result.png"

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_combo_node.get_default_gpt_image_edit_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_combo_node.bytesio_to_image_tensor",
        lambda _image_bytesio: torch.ones((1, 2, 2, 3), dtype=torch.float32),
    )

    image, output_text, result_url = node._single_request(
        "edit image",
        [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "3:4",
        "2K",
        "high",
        False,
        "req-resize",
    )

    assert tuple(image.shape) == (1, 2, 2, 3)
    assert output_text == "ok"
    assert result_url == "https://example.com/result.png"
    assert captured["resize"] is False
    assert captured["aspect_ratio"] == "3:4"
    assert captured["quality"] == "high"
    assert captured["out_request_id"] == "req-resize"


def test_gpt_combo_size_override_passed_to_client(monkeypatch):
    node = FD_GPTImageComboNode()
    captured = {}

    class FakeClient:
        def edit_image(self, **kwargs):
            captured.update(kwargs)
            return io.BytesIO(b"fake-image"), "ok", "https://example.com/result.png"

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_combo_node.get_default_gpt_image_edit_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_image_combo_node.bytesio_to_image_tensor",
        lambda _image_bytesio: torch.ones((1, 2, 2, 3), dtype=torch.float32),
    )

    node._single_request(
        "edit image",
        [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "3:4",
        "2K",
        "high",
        False,
        "req-override",
        size_override="1537x1025",
    )

    assert captured["size"] == "1537x1025"
    assert captured["aspect_ratio"] == ""


def test_gpt_multi_single_request_passes_resize_to_client(monkeypatch):
    node = FD_GPTMultiImage()
    captured = {}

    class FakeClient:
        def edit_image(self, **kwargs):
            captured.update(kwargs)
            return io.BytesIO(b"fake-image"), "ok", "https://example.com/result.png"

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_multi_image_node.get_default_gpt_image_edit_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_multi_image_node.bytesio_to_image_tensor",
        lambda _image_bytesio: torch.ones((1, 2, 2, 3), dtype=torch.float32),
    )

    image = node._single_request(
        "edit image",
        [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "3:4",
        "2K",
        "high",
        False,
        "req-resize",
    )

    assert tuple(image.shape) == (1, 2, 2, 3)
    assert captured["resize"] is False
    assert captured["aspect_ratio"] == "3:4"
    assert captured["quality"] == "high"
    assert captured["out_request_id"] == "req-resize"


def test_gpt_multi_size_override_passed_to_client(monkeypatch):
    node = FD_GPTMultiImage()
    captured = {}

    class FakeClient:
        def edit_image(self, **kwargs):
            captured.update(kwargs)
            return io.BytesIO(b"fake-image"), "ok", "https://example.com/result.png"

    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_multi_image_node.get_default_gpt_image_edit_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.gpt_multi_image_node.bytesio_to_image_tensor",
        lambda _image_bytesio: torch.ones((1, 2, 2, 3), dtype=torch.float32),
    )

    node._single_request(
        "edit image",
        [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "16:9",
        "2K",
        "high",
        False,
        "req-override",
        size_override="2048x2048",
    )

    assert captured["size"] == "2048x2048"
    assert captured["aspect_ratio"] == ""


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


def test_gemini_image_nodes_append_color_bias_and_batch_inputs():
    for node_cls in (ZhiYiImageToImageNode, ZhiYiImageToImageComboNode, FD_GeminiImage):
        optional_inputs = node_cls.INPUT_TYPES()["optional"]
        assert list(optional_inputs)[-4:] == [
            "enable_color_bias_correction",
            "color_bias_reference_image_index",
            "batch_task_id",
            "batch_item_id",
        ]
        assert optional_inputs["enable_color_bias_correction"][1]["default"] is False
        assert optional_inputs["color_bias_reference_image_index"][1]["default"] == 0
        assert optional_inputs["batch_task_id"][1]["default"] == ""
        assert optional_inputs["batch_item_id"][1]["default"] == ""

    signatures = [
        inspect.signature(FD_GeminiImage.api_call),
        inspect.signature(ZhiYiImageToImageNode.generate),
        inspect.signature(ZhiYiImageToImageComboNode.generate),
    ]
    for signature in signatures:
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.name != "self"
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        ]
        assert [parameter.name for parameter in parameters[-4:]] == [
            "enable_color_bias_correction",
            "color_bias_reference_image_index",
            "batch_task_id",
            "batch_item_id",
        ]
        assert signature.parameters["enable_color_bias_correction"].default is False
        assert signature.parameters["color_bias_reference_image_index"].default == 0
        assert signature.parameters["batch_task_id"].default == ""
        assert signature.parameters["batch_item_id"].default == ""


def test_zhiyi_image_to_image_nodes_expose_legacy_and_channel_models():
    expected_models = {
        "google/gemini-2.5-flash-image-preview",
        "google/gemini-3-pro-image-preview",
        "google/gemini-3-pro-image-preview-stable",
        "google/gemini-3-pro-image-preview-cheap",
        "gemini-3-pro-image-preview-stable",
        "gemini-3-pro-image-preview-cheap",
        "google/gemini-3-pro-image-preview-official",
        "google/gemini-3.1-flash-image-preview",
        "batch/gemini-3-pro-image-preview",
        "batch/gemini-3-pro-image-preview-stable",
        "batch/gemini-3-pro-image-preview-cheap",
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview-aistudio",
        "gemini-3-pro-image-preview-siphonlab",
        "gemini-3-pro-image-preview-vip",
        "google/gemini-3-pro-image-preview-vip",
        "batch/gemini-3-pro-image-preview-vip",
        "gemini-3-pro-image-preview-adobe",
    }

    assert expected_models.issubset(set(ZhiYiImageToImageNode.MODELS))
    assert expected_models.issubset(set(ZhiYiImageToImageComboNode.MODELS))
    new_models = {
        "google/gemini-3-pro-image-preview-stable",
        "google/gemini-3-pro-image-preview-cheap",
        "gemini-3-pro-image-preview-stable",
        "gemini-3-pro-image-preview-cheap",
        "batch/gemini-3-pro-image-preview-stable",
        "batch/gemini-3-pro-image-preview-cheap",
    }
    for model in new_models:
        assert ZhiYiImageToImageNode.MODELS.count(model) == 1
        assert ZhiYiImageToImageComboNode.MODELS.count(model) == 1
    fd_models = FD_GeminiImage.INPUT_TYPES()["required"]["model"][1]["options"]
    assert "batch/gemini-3-pro-image-preview" in fd_models
    vip_models = {
        "gemini-3-pro-image-preview-vip",
        "google/gemini-3-pro-image-preview-vip",
        "batch/gemini-3-pro-image-preview-vip",
    }
    assert vip_models.issubset(set(fd_models))
    assert fd_models.count("gemini-3-pro-image-preview-adobe") == 1
    assert ZhiYiImageToImageNode.MODELS.count("gemini-3-pro-image-preview-adobe") == 1
    assert ZhiYiImageToImageComboNode.MODELS.count("gemini-3-pro-image-preview-adobe") == 1
    for model in new_models:
        assert fd_models.count(model) == 1
    assert ZhiYiImageToImageNode.INPUT_TYPES()["required"]["model"][1]["default"] == "google/gemini-3-pro-image-preview"
    assert ZhiYiImageToImageComboNode.INPUT_TYPES()["required"]["model"][1]["default"] == "google/gemini-3-pro-image-preview"
    assert FD_GeminiImage.INPUT_TYPES()["required"]["model"][1]["default"] == GeminiImageModel.gemini_2_5_flash_image_preview


def test_fd_gemini_image_sends_color_bias_fields_as_json_types(monkeypatch):
    node = FD_GeminiImage()
    requests_bodies = []

    def fake_post(_url, json):
        requests_bodies.append(json)
        raise RuntimeError("stop after request capture")

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.old_gemini_api_node.requests.post", fake_post)

    with pytest.raises(RuntimeError, match="stop after request capture"):
        node.api_call(
            out_request_id="req-color",
            prompt="generate",
            model="google/gemini-3-pro-image-preview",
            resolution="2K",
            enable_color_bias_correction=True,
            color_bias_reference_image_index=2,
        )

    assert requests_bodies[0]["enable_color_bias_correction"] is True
    assert requests_bodies[0]["color_bias_reference_image_index"] == 2

    with pytest.raises(RuntimeError, match="stop after request capture"):
        node.api_call(
            out_request_id="req-disabled",
            prompt="generate",
            model="google/gemini-3-pro-image-preview",
            resolution="2K",
            enable_color_bias_correction="true",
            color_bias_reference_image_index=2,
        )

    assert "enable_color_bias_correction" not in requests_bodies[1]
    assert "color_bias_reference_image_index" not in requests_bodies[1]


def test_fd_gemini_image_batch_multi_image_uses_one_item_id(monkeypatch):
    uploads = []
    request_bodies = []
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.old_gemini_api_node.upload_bytes_to_oss",
        lambda path, data: uploads.append((path, data)) or f"https://oss/{len(uploads)}.png",
    )

    def fake_post(_url, json):
        request_bodies.append(json)
        raise RuntimeError("stop after request capture")

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.old_gemini_api_node.requests.post", fake_post)
    node = FD_GeminiImage()

    with pytest.raises(RuntimeError, match="stop after request capture"):
        node.api_call(
            out_request_id="req-batch",
            prompt="generate",
            model="batch/gemini-3-pro-image-preview-vip",
            images=torch.zeros((2, 2, 2, 3)),
            batch_task_id="task-1",
            batch_item_id="item-1",
        )

    assert len(uploads) == 2
    assert len(request_bodies) == 1
    assert request_bodies[0]["batch_task_id"] == "task-1"
    assert request_bodies[0]["batch_item_id"] == "item-1"
    assert request_bodies[0]["model"] == "batch/gemini-3-pro-image-preview-vip"
    assert len(request_bodies[0]["image_url_list"]) == 2


def test_fd_gemini_image_batch_validation_precedes_upload(monkeypatch):
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.old_gemini_api_node.upload_bytes_to_oss",
        lambda *_args: pytest.fail("must validate before upload"),
    )

    with pytest.raises(RuntimeError, match="batch_item_id"):
        FD_GeminiImage().api_call(
            out_request_id="req-batch",
            prompt="generate",
            model="batch/gemini-3-pro-image-preview",
            images=torch.zeros((1, 2, 2, 3)),
            batch_task_id="task",
            batch_item_id=" ",
        )


def test_gemini_service_builds_internal_request_body():
    client = GeminiImageServiceClient(service_url="https://gemini.internal")

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

    enabled_body = client.build_request_body(
        prompt="draw product",
        model="gemini-3-pro-image-preview",
        image_url_list=["https://oss/model.png", "https://oss/clothes.png"],
        enable_color_bias_correction=True,
        color_bias_reference_image_index=1,
    )
    assert enabled_body["enable_color_bias_correction"] is True
    assert enabled_body["color_bias_reference_image_index"] == 1
    assert isinstance(enabled_body["enable_color_bias_correction"], bool)

    for disabled_value in (False, "true", 1, None):
        disabled_body = client.build_request_body(
            prompt="draw product",
            model="gemini-3-pro-image-preview",
            image_url_list=["https://oss/input.png"],
            enable_color_bias_correction=disabled_value,
            color_bias_reference_image_index=2,
        )
        assert "enable_color_bias_correction" not in disabled_body
        assert "color_bias_reference_image_index" not in disabled_body

    for invalid_index in (True, "2", 2.0, None):
        normalized_body = client.build_request_body(
            prompt="draw product",
            model="gemini-3-pro-image-preview",
            image_url_list=["https://oss/input.png"],
            enable_color_bias_correction=True,
            color_bias_reference_image_index=invalid_index,
        )
        assert normalized_body["color_bias_reference_image_index"] == 0

    negative_index_body = client.build_request_body(
        prompt="draw product",
        model="gemini-3-pro-image-preview",
        image_url_list=["https://oss/input.png"],
        enable_color_bias_correction=True,
        color_bias_reference_image_index=-3,
    )
    assert negative_index_body["color_bias_reference_image_index"] == -3

    old_batch_body = client.build_request_body(
        prompt="draw product",
        model="batch/gemini-3-pro-image-preview",
        image_url_list=["https://oss/input.png"],
        batch_task_id="task-old",
        batch_item_id="item-old",
    )
    assert old_batch_body["model"] == "batch/gemini-3-pro-image-preview"
    assert old_batch_body["batch_task_id"] == "task-old"
    assert old_batch_body["batch_item_id"] == "item-old"

    new_models = [
        "google/gemini-3-pro-image-preview-stable",
        "google/gemini-3-pro-image-preview-cheap",
        "batch/gemini-3-pro-image-preview-stable",
        "batch/gemini-3-pro-image-preview-cheap",
    ]
    for model in new_models:
        assert normalize_gemini_model_name(model) == model
        body = client.build_request_body(
            prompt="draw product",
            model=model,
            image_url_list=["https://oss/input.png"],
            batch_task_id="task" if model.startswith("batch/") else "",
            batch_item_id="item" if model.startswith("batch/") else "",
        )
        assert body["model"] == model
        assert should_use_litellm_gemini(model) is False

    aliases = {
        "gemini-3-pro-image-preview-stable": "google/gemini-3-pro-image-preview-stable",
        "gemini-3-pro-image-preview-cheap": "google/gemini-3-pro-image-preview-cheap",
    }
    for alias, normalized in aliases.items():
        assert normalize_gemini_model_name(alias) == normalized
        body = client.build_request_body(
            prompt="draw product",
            model=alias,
            image_url_list=["https://oss/input.png"],
        )
        assert body["model"] == normalized
        assert should_use_litellm_gemini(alias) is False

    aistudio_body = client.build_request_body(
        prompt="draw product",
        model="gemini-3-pro-image-preview-aistudio",
        image_url_list=["https://oss/input.png"],
    )
    assert aistudio_body["model"] == "google/gemini-3-pro-image-preview-official"


def test_gemini_batch_request_body_and_validation():
    client = GeminiImageServiceClient(service_url="https://gemini.internal")

    normal_body = client.build_request_body(
        prompt="draw product",
        model="gemini-3-pro-image-preview-vip",
        image_url_list=["https://oss/input.png"],
        batch_task_id="ignored-task",
        batch_item_id="ignored-item",
    )
    assert normal_body["model"] == "google/gemini-3-pro-image-preview-vip"
    assert "batch_task_id" not in normal_body
    assert "batch_item_id" not in normal_body

    batch_body = client.build_request_body(
        prompt="draw product",
        model="batch/gemini-3-pro-image-preview-vip",
        image_url_list=["https://oss/input.png"],
        enable_color_bias_correction=True,
        color_bias_reference_image_index=0,
        batch_task_id=" task-1 ",
        batch_item_id="item-1",
    )
    assert batch_body["batch_task_id"] == " task-1 "
    assert batch_body["batch_item_id"] == "item-1"
    assert batch_body["enable_color_bias_correction"] is True
    assert isinstance(batch_body["color_bias_reference_image_index"], int)

    for missing_field, kwargs in (
        ("batch_task_id", {"batch_task_id": " ", "batch_item_id": "item"}),
        ("batch_item_id", {"batch_task_id": "task", "batch_item_id": "\t"}),
    ):
        with pytest.raises(RuntimeError, match=missing_field):
            client.build_request_body(
                prompt="draw product",
                model="batch/gemini-3-pro-image-preview-stable",
                image_url_list=["https://oss/input.png"],
                **kwargs,
            )


def test_gemini_client_batch_validation_precedes_upload_and_http(monkeypatch):
    uploads = []
    posts = []
    client = GeminiImageServiceClient(
        service_url="https://gemini.internal",
        oss_uploader=lambda path, data: uploads.append((path, data)) or "https://oss/input.png",
    )
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.utils.gemini_service.requests.post",
        lambda *args, **kwargs: posts.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="batch_item_id"):
        client.call(
            prompt="draw product",
            model="batch/gemini-3-pro-image-preview-vip",
            image_tensors=[torch.zeros((1, 2, 2, 3))],
            batch_task_id="task",
            batch_item_id="",
        )
    assert uploads == []

    with pytest.raises(RuntimeError, match="batch_task_id"):
        client.call_with_image_urls(
            prompt="draw product",
            model="batch/gemini-3-pro-image-preview",
            image_url_list=["https://oss/input.png"],
            batch_task_id=" ",
            batch_item_id="item",
        )
    assert posts == []


def test_gemini_adobe_request_body_preserves_model_and_color_bias():
    client = GeminiImageServiceClient(service_url="https://gemini.internal")
    body = client.build_request_body(
        prompt="draw product",
        model="gemini-3-pro-image-preview-adobe",
        image_url_list=["https://oss/input.png"],
        enable_color_bias_correction=True,
        color_bias_reference_image_index=2,
    )

    assert body["model"] == "gemini-3-pro-image-preview-adobe"
    assert body["enable_color_bias_correction"] is True
    assert body["color_bias_reference_image_index"] == 2
    assert "batch_task_id" not in body
    assert "batch_item_id" not in body


def test_gemini_service_model_and_prompt_helpers():
    assert normalize_gemini_model_name("gemini-2.5-flash-image-preview") == "google/gemini-2.5-flash-image-preview"
    assert normalize_gemini_model_name("google/gemini-3-pro-image-preview") == "google/gemini-3-pro-image-preview"
    assert normalize_gemini_model_name("batch/gemini-3-pro-image-preview") == "batch/gemini-3-pro-image-preview"
    assert normalize_gemini_model_name("gemini-3-pro-image-preview-vip") == "google/gemini-3-pro-image-preview-vip"
    assert normalize_gemini_model_name("google/gemini-3-pro-image-preview-vip") == "google/gemini-3-pro-image-preview-vip"
    assert normalize_gemini_model_name("batch/gemini-3-pro-image-preview-vip") == "batch/gemini-3-pro-image-preview-vip"
    assert normalize_gemini_model_name("gemini-3-pro-image-preview-adobe") == "gemini-3-pro-image-preview-adobe"
    assert normalize_gemini_model_name("gemini-3-pro-image-preview-official") == "google/gemini-3-pro-image-preview-official"
    assert normalize_gemini_model_name("google/gemini-3-pro-image-preview-official") == "google/gemini-3-pro-image-preview-official"
    assert normalize_gemini_model_name("gemini-3-pro-image-preview-aistudio") == "google/gemini-3-pro-image-preview-official"
    assert should_use_litellm_gemini("gemini-3-pro-image-preview-aistudio") is False
    assert should_use_litellm_gemini("batch/gemini-3-pro-image-preview") is False
    assert should_use_litellm_gemini("gemini-3-pro-image-preview-vip") is False
    assert should_use_litellm_gemini("google/gemini-3-pro-image-preview-vip") is False
    assert should_use_litellm_gemini("batch/gemini-3-pro-image-preview-vip") is False
    assert should_use_litellm_gemini("gemini-3-pro-image-preview-adobe") is False
    assert should_use_litellm_gemini("gemini-3-pro-image-preview-siphonlab") is True
    assert should_use_litellm_gemini("gemini-3.1-flash-image-preview") is False
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
        enable_color_bias_correction=True,
        color_bias_reference_image_index=1,
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
    assert request["enable_color_bias_correction"] is True
    assert request["color_bias_reference_image_index"] == 1
    assert request["batch_task_id"] == ""
    assert request["batch_item_id"] == ""


def test_zhiyi_image_to_image_derives_batch_item_ids_in_prompt_batch_order(monkeypatch):
    node = ZhiYiImageToImageNode()
    calls = []

    class FakeClient:
        def upload_images(self, _image_tensors):
            return ["https://oss/input.png"]

        def call_with_image_urls(self, **kwargs):
            calls.append(kwargs)
            return torch.ones((1, 2, 2, 3)), "https://oss/result.png", "ok"

    node.gemini_client = FakeClient()
    result, _seed = node.generate(
        image_1=torch.zeros((1, 2, 2, 3)),
        prompt="fallback",
        prompt_list=["prompt one", "prompt two"],
        model="batch/gemini-3-pro-image-preview-vip",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=2,
        seed_mode="固定种子",
        seed=7,
        batch_task_id="task-1",
        batch_item_id="item",
    )

    assert result.shape[0] == 4
    assert [call["prompt"] for call in calls] == [
        "prompt one",
        "prompt one",
        "prompt two",
        "prompt two",
    ]
    assert [call["batch_item_id"] for call in calls] == [
        "item-0",
        "item-1",
        "item-2",
        "item-3",
    ]
    assert {call["batch_task_id"] for call in calls} == {"task-1"}


def test_zhiyi_image_to_image_batch_validation_precedes_upload():
    node = ZhiYiImageToImageNode()
    node.gemini_client.upload_images = lambda _images: pytest.fail("must validate before upload")

    with pytest.raises(RuntimeError, match="batch_item_id"):
        node.generate(
            image_1=torch.zeros((1, 2, 2, 3)),
            prompt="prompt",
            model="batch/gemini-3-pro-image-preview",
            aspect_ratio="1:1",
            image_size="2K",
            batch_task_id="task",
            batch_item_id=" ",
        )


def test_zhiyi_image_to_image_aistudio_uses_internal_official_model(monkeypatch):
    node = ZhiYiImageToImageNode()
    calls = []

    class FakeClient:
        def upload_images(self, image_tensors):
            calls.append(("upload_images", len(image_tensors)))
            return ["https://oss/input.png"]

        def call_with_image_urls(self, **kwargs):
            calls.append(("call", kwargs))
            body = GeminiImageServiceClient(service_url="https://gemini.internal").build_request_body(**kwargs)
            calls.append(("body", body))
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
        enable_color_bias_correction=True,
        color_bias_reference_image_index=2,
    )

    assert result.shape == (1, 2, 2, 3)
    assert actual_seed == 42
    assert calls[0] == ("upload_images", 1)
    request = calls[1][1]
    assert request["image_url_list"] == ["https://oss/input.png"]
    assert request["prompt"] == "system prompt\n\nuser prompt"
    assert request["model"] == "gemini-3-pro-image-preview-aistudio"
    assert request["enable_color_bias_correction"] is True
    assert request["color_bias_reference_image_index"] == 2
    assert normalize_gemini_model_name(request["model"]) == "google/gemini-3-pro-image-preview-official"
    body = calls[2][1]
    assert body["enable_color_bias_correction"] is True
    assert body["color_bias_reference_image_index"] == 2


def test_zhiyi_image_to_image_flash_uses_internal_image_server(monkeypatch):
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
    monkeypatch.setattr(node, "_single_litellm_request", lambda *args: pytest.fail("flash image server model should not use LiteLLM directly"))

    result, actual_seed = node.generate(
        image_1=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        prompt="user prompt",
        model="gemini-3.1-flash-image-preview",
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
    assert request["model"] == "gemini-3.1-flash-image-preview"
    assert normalize_gemini_model_name(request["model"]) == "google/gemini-3.1-flash-image-preview"


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
        enable_color_bias_correction=True,
        color_bias_reference_image_index=1,
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
        True,
        1,
        "",
        "",
    )
    assert calls[1][1] == "system prompt\n\nprompt two"




def test_zhiyi_image_to_image_combo_batch_ids_prefer_explicit_values(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    calls = []
    combo = {
        "images": [torch.zeros((1, 2, 2, 3))],
        "prompts": ["prompt"],
    }

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _images: ["https://oss/input.png"])
    monkeypatch.setattr(
        node,
        "_single_request",
        lambda *args: calls.append(args) or torch.ones((1, 2, 2, 3)),
    )

    node.generate(
        model="batch/gemini-3-pro-image-preview-vip",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="legacy-request",
        combo_1=combo,
        batch_task_id=" explicit-task ",
        batch_item_id=" explicit-item ",
    )

    assert calls[0][-2:] == (" explicit-task ", " explicit-item ")


def test_zhiyi_image_to_image_combo_batch_ids_fall_back_to_out_request_id(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    calls = []
    combo = {
        "images": [torch.zeros((1, 2, 2, 3))],
        "prompts": ["prompt"],
    }

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _images: ["https://oss/input.png"])
    monkeypatch.setattr(
        node,
        "_single_request",
        lambda *args: calls.append(args) or torch.ones((1, 2, 2, 3)),
    )

    node.generate(
        model="batch/gemini-3-pro-image-preview-vip",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="legacy-request",
        combo_1=combo,
    )

    assert calls[0][-2:] == ("legacy-request", "legacy-request")


def test_zhiyi_image_to_image_combo_batch_item_fallback_uses_effective_task_id(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    calls = []
    combo = {
        "images": [torch.zeros((1, 2, 2, 3))],
        "prompts": ["prompt one", "prompt two"],
    }

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _images: ["https://oss/input.png"])
    monkeypatch.setattr(
        node,
        "_single_request",
        lambda *args: calls.append(args) or torch.ones((1, 2, 2, 3)),
    )

    node.generate(
        model="batch/gemini-3-pro-image-preview-vip",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="legacy-request",
        combo_1=combo,
    )
    assert [args[-2:] for args in calls] == [
        ("legacy-request", "legacy-request-0"),
        ("legacy-request", "legacy-request-1"),
    ]

    calls.clear()
    node.generate(
        model="batch/gemini-3-pro-image-preview-vip",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="legacy-request",
        combo_1={"images": [torch.zeros((1, 2, 2, 3))], "prompts": ["prompt one", "prompt two"]},
        batch_task_id="explicit-task",
    )
    assert [args[-2:] for args in calls] == [
        ("explicit-task", "explicit-task-0"),
        ("explicit-task", "explicit-task-1"),
    ]


def test_zhiyi_image_to_image_combo_batch_sync_response_does_not_reuse_server_task_id(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    node.gemini_client.service_url = "https://gemini.internal/generate"
    combo = {
        "images": [torch.zeros((1, 2, 2, 3))],
        "prompts": ["prompt"],
    }
    posts = []
    gets = []
    image_buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="white").save(image_buffer, format="PNG")

    class FakeResponse:
        ok = True
        status_code = 200
        text = "{}"
        content = image_buffer.getvalue()

        def json(self):
            return {
                "result_image_url": "https://oss/result.png",
                "task_id": "server-diagnostic-task",
            }

        def raise_for_status(self):
            return None

    def fake_post(_url, json, timeout):
        posts.append((json, timeout))
        return FakeResponse()

    def fake_get(url, timeout):
        gets.append((url, timeout))
        return FakeResponse()

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.utils.gemini_service.requests.post", fake_post)
    monkeypatch.setattr("src.Comfyui_Fd_Nodes.utils.gemini_service.requests.get", fake_get)
    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _images: ["https://oss/input.png"])

    images, _seed, _log = node.generate(
        model="batch/gemini-3-pro-image-preview-vip",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="legacy-request",
        combo_1=combo,
        batch_task_id="client-task",
        batch_item_id="client-item",
    )

    assert len(images) == 1
    assert len(posts) == 1
    assert posts[0][0]["batch_task_id"] == "client-task"
    assert posts[0][0]["batch_item_id"] == "client-item"
    assert "server-diagnostic-task" not in posts[0][0].values()
    assert gets == [("https://oss/result.png", 300)]


def test_zhiyi_image_to_image_combo_non_batch_does_not_derive_batch_ids(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    calls = []
    combo = {
        "images": [torch.zeros((1, 2, 2, 3))],
        "prompts": ["prompt"],
    }

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _images: ["https://oss/input.png"])
    monkeypatch.setattr(
        node,
        "_single_request",
        lambda *args: calls.append(args) or torch.ones((1, 2, 2, 3)),
    )

    node.generate(
        model="gemini-3-pro-image-preview-vip",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="legacy-request",
        combo_1=combo,
    )

    assert calls[0][-2:] == ("", "")

    node = ZhiYiImageToImageComboNode()
    valid_combo = {
        "images": [torch.zeros((1, 2, 2, 3))],
        "prompts": ["prompt one", "prompt two"],
    }
    invalid_combo = {"images": [], "prompts": ["skipped"]}
    calls = []

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _images: ["https://oss/input.png"])

    def fake_single_request(*args):
        calls.append(args)
        return torch.ones((1, 2, 2, 3))

    monkeypatch.setattr(node, "_single_request", fake_single_request)
    images, _seed, _log = node.generate(
        model="batch/gemini-3-pro-image-preview-vip",
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        combo_1=invalid_combo,
        combo_2=valid_combo,
        batch_task_id="task-1",
        batch_item_id="item",
    )

    assert len(images) == 2
    assert [args[-1] for args in calls] == ["item-0", "item-1"]
    assert [args[1] for args in calls] == ["prompt one", "prompt two"]


def test_zhiyi_image_to_image_combo_batch_validation_precedes_preprocessing(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    node.gemini_client.upload_images = lambda _images: pytest.fail("must validate before upload")
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.utils.gemini_service.requests.post",
        lambda *_args, **_kwargs: pytest.fail("must validate before HTTP"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"提交阶段.*batch_task_id.*out_request_id",
    ):
        node.generate(
            model="batch/gemini-3-pro-image-preview-vip",
            aspect_ratio="1:1",
            image_size="4K",
            out_request_id=" ",
            combo_1={"images": [torch.zeros((1, 2, 2, 3))], "prompts": ["prompt"]},
            batch_task_id="",
            batch_item_id="item",
        )


def test_zhiyi_image_to_image_combo_aistudio_uses_internal_official_model(monkeypatch):
    node = ZhiYiImageToImageComboNode()
    node.gemini_client.service_url = "https://gemini.internal/generate"
    combo_1 = {
        "images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "prompts": ["prompt one"],
    }
    combo_2 = {
        "images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "prompts": ["prompt two"],
    }
    calls = []

    monkeypatch.setattr(node.gemini_client, "upload_images", lambda _images: ["https://oss/input.png"])
    monkeypatch.setattr(node, "_single_litellm_request", lambda *args: pytest.fail("aistudio should use internal Gemini service"))

    def fake_single_request(*args):
        calls.append(args)
        body = GeminiImageServiceClient(service_url="https://gemini.internal").build_request_body(
            prompt=args[1],
            model=args[2],
            image_url_list=args[0],
            aspect_ratio=args[3],
            image_size=args[4],
            out_request_id=args[6],
            enable_color_bias_correction=args[7],
            color_bias_reference_image_index=args[8],
            batch_task_id=args[9],
            batch_item_id=args[10],
        )
        calls.append(body)
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
        combo_1=combo_1,
        combo_2=combo_2,
        system_prompt="system prompt",
        enable_color_bias_correction=True,
        color_bias_reference_image_index=2,
    )

    assert len(images) == 2
    assert actual_seed == 7
    assert "url=https://gemini.internal/generate" in log_text
    request_args = [call for call in calls if isinstance(call, tuple)]
    bodies = [call for call in calls if isinstance(call, dict)]
    assert len(request_args) == 2
    assert len(bodies) == 2
    for args, body in zip(request_args, bodies):
        assert args[2] == "gemini-3-pro-image-preview-aistudio"
        assert args[7] is True
        assert args[8] == 2
        assert body["enable_color_bias_correction"] is True
        assert body["color_bias_reference_image_index"] == 2
        assert body["model"] == "google/gemini-3-pro-image-preview-official"


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3-pro-image-preview-siphonlab",
    ],
)
def test_zhiyi_image_to_image_combo_siphonlab_uses_litellm_without_model_rename(monkeypatch, model):
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
        model=model,
        aspect_ratio="1:1",
        image_size="4K",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        out_request_id="req-456",
        combo_1=combo,
        system_prompt="system prompt",
        enable_color_bias_correction=True,
        color_bias_reference_image_index=1,
    )

    assert len(images) == 1
    assert actual_seed == 7
    assert "FD_LITELLM_BASE_URL/v1/chat/completions" in log_text
    assert calls[0][1] == model
    assert calls[0][0][0]["role"] == "system"
    assert calls[0][0][1]["content"][0]["text"] == "prompt one"
    assert calls[0][0][1]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_zhiyi_image_text_request_logs_request_and_response_without_images(monkeypatch):
    """The image-text API should log request/response summaries without image payloads."""
    node = ZhiYiImageTextNode()
    captured_logs = []

    class DummyResponse:
        ok = True
        status_code = 200
        text = '{"choices":[{"message":{"content":"hello"}}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

    def fake_post(url, headers, data, timeout):
        captured_logs.append(("post_payload", json.loads(data)))
        return DummyResponse()

    def fake_image_tensor_to_data_url(_image, max_data_url_bytes):
        return (
            "data:image/jpeg;base64,AAA",
            {
                "original_size": (1, 1),
                "final_size": (1, 1),
                "original_pixels": 1,
                "final_pixels": 1,
                "max_total_pixels": zhiyi_image_text_module.MAX_IMAGE_TOTAL_PIXELS,
                "resized_by_pixels": False,
                "mime_type": "image/jpeg",
                "quality": 85,
                "image_bytes": 3,
                "data_url_bytes": 26,
                "max_data_url_bytes": max_data_url_bytes,
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
    assert request_log["request_body_bytes"] <= zhiyi_image_text_module.MAX_REQUEST_BODY_BYTES
    assert request_log["max_request_body_bytes"] == zhiyi_image_text_module.MAX_REQUEST_BODY_BYTES
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
    assert info["original_pixels"] == 256
    assert info["final_pixels"] == 256
    assert info["max_total_pixels"] == zhiyi_image_text_module.MAX_IMAGE_TOTAL_PIXELS
    assert info["resized_by_pixels"] is False
    assert info["quality"] == 85
    assert info["resized"] is False


def test_zhiyi_image_text_resize_to_max_pixels_preserves_ratio_without_upscaling():
    """Images above the model pixel limit should be downscaled before encoding."""
    node = ZhiYiImageTextNode()
    large_image = Image.new("RGB", (5386, 7137))

    resized = node._resize_to_max_pixels(large_image, zhiyi_image_text_module.MAX_IMAGE_TOTAL_PIXELS)

    width, height = resized.size
    assert width * height <= zhiyi_image_text_module.MAX_IMAGE_TOTAL_PIXELS
    assert width < 5386
    assert height < 7137
    assert abs((width / height) - (5386 / 7137)) < 0.001

    small_image = Image.new("RGB", (100, 80))
    assert node._resize_to_max_pixels(small_image, zhiyi_image_text_module.MAX_IMAGE_TOTAL_PIXELS) is small_image


def test_zhiyi_image_text_data_url_info_tracks_pixel_resize(monkeypatch):
    """Encoded image info should expose pixel-limit resizing details."""
    node = ZhiYiImageTextNode()
    image = torch.zeros((1, 200, 200, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_module, "MAX_IMAGE_TOTAL_PIXELS", 10_000)

    data_url, info = node._image_tensor_to_data_url(image)

    assert data_url.startswith("data:image/jpeg;base64,")
    assert info["original_size"] == (200, 200)
    assert info["original_pixels"] == 40_000
    assert info["final_pixels"] <= 10_000
    assert info["max_total_pixels"] == 10_000
    assert info["resized_by_pixels"] is True
    assert info["resized"] is True


def test_zhiyi_image_text_recompresses_until_data_url_is_under_limit(monkeypatch):
    """Large/noisy inputs should be retried with lower quality and smaller dimensions until under the transport limit."""
    node = ZhiYiImageTextNode()
    image = torch.rand((1, 512, 512, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_module, "MIN_IMAGE_LONG_EDGE", 128)
    monkeypatch.setattr(zhiyi_image_text_module, "IMAGE_RESIZE_FACTOR", 0.5)

    data_url, info = node._image_tensor_to_data_url(
        image, max_data_url_bytes=50_000
    )

    assert data_url.startswith("data:image/jpeg;base64,")
    assert len(data_url.encode("utf-8")) <= 50_000
    assert info["data_url_bytes"] <= 50_000
    assert info["final_size"][0] <= 512
    assert info["final_size"][1] <= 512
    assert info["final_pixels"] == info["final_size"][0] * info["final_size"][1]
    assert info["max_total_pixels"] == zhiyi_image_text_module.MAX_IMAGE_TOTAL_PIXELS
    assert info["resized"] is True


def test_zhiyi_image_text_falls_back_to_original_size_when_image_cannot_fit(monkeypatch):
    """If compression cannot meet the budget, keep the original dimensions and continue."""
    node = ZhiYiImageTextNode()
    image = torch.zeros((1, 4, 6, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_module, "MIN_IMAGE_LONG_EDGE", 1)

    data_url, info = node._image_tensor_to_data_url(image, max_data_url_bytes=1)
    decoded = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))

    assert decoded.size == (6, 4)
    assert info["original_size"] == (6, 4)
    assert info["final_size"] == (6, 4)
    assert info["resized"] is False
    assert info["compression_fallback"] is True
    assert info["budget_exceeded"] is True
    assert info["data_url_bytes"] > 1


def test_zhiyi_image_text_fallback_still_sends_over_budget_request(monkeypatch):
    node = ZhiYiImageTextNode()
    captured = {}
    fallback_data_url = "data:image/jpeg;base64," + ("A" * 50_000)

    class DummyResponse:
        ok = True
        status_code = 200
        text = '{"choices":[{"message":{"content":"fallback success"}}]}'

        def json(self):
            return {"choices": [{"message": {"content": "fallback success"}}]}

    def fake_post(url, headers, data, timeout):
        captured["data"] = data
        return DummyResponse()

    monkeypatch.setattr(zhiyi_image_text_module, "load_config", lambda: {
        "base_url": "https://example.com",
        "api_key": "secret",
    })
    monkeypatch.setattr(zhiyi_image_text_module.requests, "post", fake_post)
    monkeypatch.setattr(
        node,
        "_image_tensor_to_data_url",
        lambda _image, max_data_url_bytes: (
            fallback_data_url,
            {
                "compression_fallback": True,
                "budget_exceeded": True,
                "data_url_bytes": len(fallback_data_url.encode("utf-8")),
                "max_data_url_bytes": max_data_url_bytes,
            },
        ),
    )

    result = node.generate(
        image=torch.zeros((1, 2, 2, 3)),
        prompt="describe",
        node_switch=0,
    )

    assert result == ("fallback success",)
    assert len(captured["data"].encode("utf-8")) > zhiyi_image_text_module.MAX_REQUEST_BODY_BYTES
    assert fallback_data_url in captured["data"]


def test_zhiyi_image_text_fallback_preserves_upstream_http_error(monkeypatch):
    node = ZhiYiImageTextNode()
    fallback_data_url = "data:image/jpeg;base64," + ("A" * 50_000)

    class DummyResponse:
        ok = False
        status_code = 413
        text = "request entity too large"

    monkeypatch.setattr(zhiyi_image_text_module, "load_config", lambda: {
        "base_url": "https://example.com",
        "api_key": "secret",
    })
    monkeypatch.setattr(
        node,
        "_image_tensor_to_data_url",
        lambda _image, max_data_url_bytes: (
            fallback_data_url,
            {"compression_fallback": True},
        ),
    )
    monkeypatch.setattr(
        zhiyi_image_text_module.requests,
        "post",
        lambda *args, **kwargs: DummyResponse(),
    )

    with pytest.raises(RuntimeError, match="API 请求失败: 413"):
        node.generate(
            image=torch.zeros((1, 2, 2, 3)),
            prompt="describe",
            node_switch=0,
        )


def test_zhiyi_image_text_http_error_includes_response_body(monkeypatch):
    """LiteLLM 4xx errors should preserve the response body for debugging."""
    node = ZhiYiImageTextNode()

    class DummyResponse:
        ok = False
        status_code = 400
        text = '{"error":{"message":"Image exceeds max pixels"}}'

        def json(self):
            raise AssertionError("json should not be called for non-2xx responses")

    monkeypatch.setattr(zhiyi_image_text_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(
        node,
        "_image_tensor_to_data_url",
        lambda _image, max_data_url_bytes: ("data:image/jpeg;base64,AAA", {}),
    )
    monkeypatch.setattr(zhiyi_image_text_module.requests, "post", lambda *args, **kwargs: DummyResponse())

    with pytest.raises(RuntimeError) as exc_info:
        node.generate(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            prompt="describe this image",
            node_switch=0,
        )

    message = str(exc_info.value)
    assert "API 请求失败: 400" in message
    assert "Image exceeds max pixels" in message


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

    encoded_count = 0

    def fake_image_tensor_to_data_url(image_tensor, max_data_url_bytes):
        nonlocal encoded_count
        encoded_count += 1
        index = encoded_count
        return (
            f"data:image/jpeg;base64,IMG{index}",
            {
                "original_size": (2, 2),
                "final_size": (2, 2),
                "mime_type": "image/jpeg",
                "quality": 85,
                "image_bytes": index,
                "data_url_bytes": 30,
                "max_data_url_bytes": max_data_url_bytes,
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


def test_zhiyi_image_text_combo_fallback_still_sends_over_budget_request(monkeypatch):
    node = ZhiYiImageTextComboNode()
    captured = {}
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    fallback_data_url = "data:image/jpeg;base64," + ("A" * 50_000)

    class DummyResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"fallback combo"}}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "fallback combo"}}]}

    def fake_post(url, headers, data, timeout):
        captured["data"] = data
        return DummyResponse()

    monkeypatch.setattr(zhiyi_image_text_combo_module, "load_config", lambda: {
        "base_url": "https://example.com",
        "api_key": "secret",
    })
    monkeypatch.setattr(zhiyi_image_text_combo_module.requests, "post", fake_post)
    monkeypatch.setattr(
        node,
        "_image_tensor_to_data_url",
        lambda _image, max_data_url_bytes: (
            fallback_data_url,
            {
                "compression_fallback": True,
                "budget_exceeded": True,
                "data_url_bytes": len(fallback_data_url.encode("utf-8")),
                "max_data_url_bytes": max_data_url_bytes,
            },
        ),
    )

    text, _log = node.generate(
        max_concurrency=1,
        retry_count=0,
        combo_1={"images": [image], "prompts": ["describe"]},
    )

    assert text == "fallback combo"
    assert len(captured["data"].encode("utf-8")) > zhiyi_image_text_module.MAX_REQUEST_BODY_BYTES
    assert fallback_data_url in captured["data"]


def test_zhiyi_image_text_combo_keeps_output_order_when_requests_finish_out_of_order(monkeypatch):
    node = ZhiYiImageTextComboNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_combo_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(
        node,
        "_image_tensor_to_data_url",
        lambda _image, max_data_url_bytes: ("data:image/jpeg;base64,AAA", {}),
    )

    def fake_single_request(url, api_key, request_body, request_log, retry_count):
        prompt = json.loads(request_body)["messages"][-1]["content"][0]["text"]
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

    request_body = json.dumps({
        "stream": False,
        "model": "doubao-seed-2.0-mini",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}],
        "temperature": 0.7,
        "max_tokens": 128,
    })
    text, retry_used = node._single_request(
        url="https://example.com/v1/chat/completions",
        api_key="secret",
        request_body=request_body,
        request_log={},
        retry_count=1,
    )

    assert text == "retry success"
    assert retry_used == 1
    assert len(calls) == 2


def test_zhiyi_image_text_combo_keeps_failure_placeholder_for_partial_failure(monkeypatch):
    node = ZhiYiImageTextComboNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(zhiyi_image_text_combo_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(
        node,
        "_image_tensor_to_data_url",
        lambda _image, max_data_url_bytes: ("data:image/jpeg;base64,AAA", {}),
    )

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
    monkeypatch.setattr(
        node,
        "_image_tensor_to_data_url",
        lambda _image, max_data_url_bytes: ("data:image/jpeg;base64,AAA", {}),
    )
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

def test_zhiyi_image_text_large_image_request_stays_under_48kb_and_is_valid_jpeg(monkeypatch):
    node = ZhiYiImageTextNode()
    image = torch.rand((1, 2730, 2048, 3), dtype=torch.float32)
    captured = {"logs": []}
    original_serialize_request_body = node._serialize_request_body

    class DummyResponse:
        ok = True
        status_code = 200
        text = '{"choices":[{"message":{"content":"description"}}]}'

        def json(self):
            return {"choices": [{"message": {"content": "description"}}]}

    def fake_post(url, headers, data, timeout):
        captured.update(url=url, headers=headers, data=data, timeout=timeout)
        return DummyResponse()

    def capture_serialized_body(payload, allow_over_budget=False):
        result = original_serialize_request_body(
            payload, allow_over_budget=allow_over_budget
        )
        captured["serialized_body"] = result[0]
        return result

    monkeypatch.setattr(zhiyi_image_text_module, "load_config", lambda: {
        "base_url": "https://example.com",
        "api_key": "secret",
    })
    monkeypatch.setattr(zhiyi_image_text_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_serialize_request_body", capture_serialized_body)
    monkeypatch.setattr(
        zhiyi_image_text_module.logger,
        "info",
        lambda message, *args: captured["logs"].append((message, args)),
    )

    assert node.generate(image=image, prompt="describe", node_switch=0) == ("description",)

    request_body = captured["data"]
    assert request_body is captured["serialized_body"]
    payload = json.loads(request_body)
    data_url = payload["messages"][-1]["content"][1]["image_url"]["url"]
    jpeg_bytes = base64.b64decode(data_url.split(",", 1)[1])
    decoded = Image.open(io.BytesIO(jpeg_bytes))

    assert len(request_body.encode("utf-8")) <= zhiyi_image_text_module.MAX_REQUEST_BODY_BYTES
    assert decoded.format == "JPEG"
    assert decoded.size[0] <= 2048
    assert decoded.size[1] <= 2730
    assert abs((decoded.size[0] / decoded.size[1]) - (2048 / 2730)) < 0.002
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    assert list(captured["headers"]).count("Connection") == 1
    assert captured["timeout"] == (5, 120)
    image_log = next(
        args[0]
        for message, args in captured["logs"]
        if message == "ZhiYi image-to-text encoded image: %s"
    )
    assert image_log["original_size"] == (2048, 2730)
    assert image_log["final_size"] == decoded.size
    assert image_log["quality"] in zhiyi_image_text_module.JPEG_QUALITY_STEPS
    assert image_log["image_bytes"] == len(jpeg_bytes)
    assert image_log["data_url_bytes"] == len(data_url.encode("utf-8"))
    assert image_log["request_body_bytes"] == len(request_body.encode("utf-8"))
    assert image_log["max_request_body_bytes"] == zhiyi_image_text_module.MAX_REQUEST_BODY_BYTES


def test_zhiyi_image_text_small_image_keeps_dimensions_with_request_budget():
    node = ZhiYiImageTextNode()
    image = torch.zeros((1, 32, 48, 3), dtype=torch.float32)
    messages = node._build_messages("describe", [""], "")
    budget = node._image_budget(node._build_request_payload(messages, 0.7, 2048), 1)

    data_url, info = node._image_tensor_to_data_url(image, max_data_url_bytes=budget)
    decoded = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))

    assert decoded.size == (48, 32)
    assert info["final_size"] == (48, 32)
    assert info["quality"] == 85
    assert info["resized"] is False


def test_zhiyi_image_text_rejects_prompt_with_no_image_budget(monkeypatch):
    node = ZhiYiImageTextNode()
    monkeypatch.setattr(zhiyi_image_text_module, "load_config", lambda: {
        "base_url": "https://example.com",
        "api_key": "secret",
    })
    monkeypatch.setattr(
        zhiyi_image_text_module.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail("must fail before HTTP"),
    )

    with pytest.raises(RuntimeError, match="没有可用的图片字节预算"):
        node.generate(
            image=torch.zeros((1, 2, 2, 3)),
            prompt="x" * zhiyi_image_text_module.MAX_REQUEST_BODY_BYTES,
            node_switch=0,
        )


def test_zhiyi_image_text_combo_multi_image_body_is_identical_and_under_48kb(monkeypatch):
    node = ZhiYiImageTextComboNode()
    captured = {}
    images = torch.zeros((2, 1024, 768, 3), dtype=torch.float32)

    class DummyResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"both"}}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "both"}}]}

    def fake_post(url, headers, data, timeout):
        captured.update(url=url, headers=headers, data=data, timeout=timeout)
        return DummyResponse()

    monkeypatch.setattr(zhiyi_image_text_combo_module, "load_config", lambda: {
        "base_url": "https://example.com",
        "api_key": "secret",
    })
    monkeypatch.setattr(zhiyi_image_text_combo_module.requests, "post", fake_post)

    text, _log = node.generate(
        max_concurrency=1,
        retry_count=0,
        combo_1={"images": [images], "prompts": ["describe both"]},
    )

    assert text == "both"
    assert len(captured["data"].encode("utf-8")) <= zhiyi_image_text_module.MAX_REQUEST_BODY_BYTES
    payload = json.loads(captured["data"])
    assert json.dumps(payload) == captured["data"]
    assert len(payload["messages"][-1]["content"]) == 3
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    assert captured["timeout"] == (5, 120)


def test_zhiyi_text_gen_request_logs_request_and_response(monkeypatch):
    """The text API should log request/response summaries."""
    node = ZhiYiTextGenNode()
    captured_logs = []

    class DummyResponse:
        ok = True
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
    assert request_payload["model"] == "doubao-seed-2.0-mini"
    assert request_payload["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "say hello"},
    ]
    request_log = next(args[0] for message, args in captured_logs if message == "Calling ZhiYi text API with payload=%s")
    assert request_log["messages"] == [
        {"role": "system", "text": "system prompt"},
        {"role": "user", "text": "say hello"},
    ]
    response_log = next(args[0] for message, args in captured_logs if message == "ZhiYi text API response summary: %s")
    assert response_log["status_code"] == 200
    assert response_log["choice_count"] == 1


def test_zhiyi_text_gen_http_error_includes_response_body(monkeypatch):
    """Text API 4xx errors should preserve the LiteLLM response body."""
    node = ZhiYiTextGenNode()

    class DummyResponse:
        ok = False
        status_code = 400
        text = '{"error":{"message":"invalid model or malformed messages"}}'

        def json(self):
            raise AssertionError("json should not be called for non-2xx responses")

    monkeypatch.setattr(zhiyi_text_module, "load_config", lambda: {"base_url": "https://example.com", "api_key": "secret"})
    monkeypatch.setattr(zhiyi_text_module.requests, "post", lambda *args, **kwargs: DummyResponse())

    with pytest.raises(RuntimeError) as exc_info:
        node.generate(
            prompt="say hello",
            node_switch=0,
            system_prompt="system prompt",
        )

    message = str(exc_info.value)
    assert "API 请求失败: 400" in message
    assert "invalid model or malformed messages" in message


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
