#!/usr/bin/env python

"""Tests for `Comfyui_Fd_Nodes` package."""

import pytest
from src.Comfyui_Fd_Nodes.gpt_image_edit_node import GPTImageEditNode
from src.Comfyui_Fd_Nodes.nodes import (
    FD_GTPImage,
    Example,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _resolution_to_edit_size,
)
from src.Comfyui_Fd_Nodes.prompt_nodes import EcommercePromptGenerator, PromptListSelector
from src.Comfyui_Fd_Nodes.zhiyi_image_text_node import ZhiYiImageTextNode
from src.Comfyui_Fd_Nodes.zhiyi_image_to_image_node import ZhiYiImageToImageNode
from src.Comfyui_Fd_Nodes.zhiyi_text_node import ZhiYiTextGenNode

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
