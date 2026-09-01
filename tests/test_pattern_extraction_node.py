"""Tests for `pattern_extraction_node` module."""

import json

import numpy as np
import torch

from src.Comfyui_Fd_Nodes import nodes
from src.Comfyui_Fd_Nodes.pattern_extraction_node import (
    PatternApplyAlphaToImage,
    PatternChooseBackgroundPair,
    PatternDualBackgroundToRGBA,
)
from src.Comfyui_Fd_Nodes.zhiyi_rmbg_segment_node import _RmbgSegmentApiBase


def _make_image_tensor(arr_uint8):
    arr = arr_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _make_subject_image(size=100):
    """深蓝底 + 中心亮黄主体块，用于触发选色逻辑。"""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:] = (10, 20, 40)
    m = size // 4
    arr[m:size - m, m:size - m] = (240, 200, 30)
    return _make_image_tensor(arr)


def _make_dual_bg_image(size=100, bg=(255, 0, 0)):
    """纯色背景 + 中心蓝色主体块（位置/尺寸固定），用于双背景差分。"""
    arr = np.full((size, size, 3), bg, dtype=np.uint8)
    m = size // 4
    arr[m:size - m, m:size - m] = (20, 30, 200)
    return _make_image_tensor(arr)


def test_choose_background_pair_returns_valid_strings():
    node = PatternChooseBackgroundPair()
    result = node.execute(_make_subject_image())

    assert len(result) == 5
    bg1_name, bg2_name, bg1_rgb_json, bg2_rgb_json, selection_json = result

    assert isinstance(bg1_name, str) and bg1_name
    assert isinstance(bg2_name, str) and bg2_name
    assert bg1_name != bg2_name

    bg1_rgb = json.loads(bg1_rgb_json)
    bg2_rgb = json.loads(bg2_rgb_json)
    assert isinstance(bg1_rgb, list) and len(bg1_rgb) == 3
    assert isinstance(bg2_rgb, list) and len(bg2_rgb) == 3

    selection = json.loads(selection_json)
    assert isinstance(selection, dict)
    assert "selection_strategy" in selection


def test_dual_background_to_rgba_outputs():
    node = PatternDualBackgroundToRGBA()
    image_a = _make_dual_bg_image(bg=(255, 0, 0))
    image_b = _make_dual_bg_image(bg=(0, 255, 0))

    result = node.execute(
        image_a, image_b, sample_span=24, alpha_floor=0.02
    )
    rgba_image, alpha_mask, meta_json, rgba_soft, rgba_agreement, preferred_rgba, preferred_alpha_mask = result

    assert len(result) == 7
    assert rgba_image.dim() == 4
    assert rgba_image.shape[0] == 1
    assert rgba_image.shape[-1] == 4
    assert alpha_mask.dim() == 3
    assert alpha_mask.shape[0] == 1

    assert rgba_soft.dim() == 4
    assert rgba_soft.shape[-1] == 4
    assert rgba_agreement.dim() == 4
    assert rgba_agreement.shape[-1] == 4
    assert preferred_rgba.dim() == 4
    assert preferred_rgba.shape[-1] == 4
    assert preferred_alpha_mask.dim() == 3
    assert preferred_alpha_mask.shape[0] == 1

    meta = json.loads(meta_json)
    assert isinstance(meta, list) and len(meta) == 1
    assert "bg_a_rgb" in meta[0]
    assert "bg_b_rgb" in meta[0]


def test_pattern_dual_background_to_rgba_return_names():
    names = PatternDualBackgroundToRGBA.RETURN_NAMES
    assert len(names) == 7
    assert names[0] == "rgba_image"
    assert names[1] == "alpha_mask"
    assert names[2] == "meta_json"
    assert names[5] == "preferred_rgba"
    assert names[6] == "preferred_alpha_mask"


def test_apply_alpha_to_image_is_registered():
    assert "PatternApplyAlphaToImage+" in nodes.NODE_CLASS_MAPPINGS
    assert nodes.NODE_CLASS_MAPPINGS["PatternApplyAlphaToImage+"] is PatternApplyAlphaToImage
    assert "PatternApplyAlphaToImage+" in nodes.NODE_DISPLAY_NAME_MAPPINGS


def test_nodes_registered_in_mappings():
    assert "PatternChooseBackgroundPair+" in nodes.NODE_CLASS_MAPPINGS
    assert "PatternDualBackgroundToRGBA+" in nodes.NODE_CLASS_MAPPINGS
    assert nodes.NODE_CLASS_MAPPINGS["PatternChooseBackgroundPair+"] is PatternChooseBackgroundPair
    assert nodes.NODE_CLASS_MAPPINGS["PatternDualBackgroundToRGBA+"] is PatternDualBackgroundToRGBA

    assert "PatternChooseBackgroundPair+" in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert "PatternDualBackgroundToRGBA+" in nodes.NODE_DISPLAY_NAME_MAPPINGS


def test_rmbg_rgba_composition_preserves_foreground_and_alpha():
    image = _make_image_tensor(np.array([
        [[200, 100, 50], [10, 20, 30]],
        [[40, 50, 60], [255, 240, 10]],
    ], dtype=np.uint8))
    mask = torch.tensor([[[1.0, 0.0], [0.5, 1.0]]])

    rgba = _RmbgSegmentApiBase()._compose_rgba_image(image, mask)

    assert rgba.shape == (1, 2, 2, 4)
    np.testing.assert_allclose(rgba[0, 0, 0].numpy(), [200 / 255, 100 / 255, 50 / 255, 1.0])
    np.testing.assert_allclose(rgba[0, 0, 1].numpy(), [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(rgba[0, 1, 0].numpy(), [40 / 255, 50 / 255, 60 / 255, 0.5])
    np.testing.assert_allclose(rgba[0, 1, 1].numpy(), [1.0, 240 / 255, 10 / 255, 1.0])


def test_rmbg_output_background_setting_controls_image_format():
    image = _make_image_tensor(np.array([[[200, 100, 50], [10, 20, 30]]], dtype=np.uint8))
    mask = torch.tensor([[[1.0, 0.0]]])
    node = _RmbgSegmentApiBase()

    alpha = node._compose_output_image(image, mask, "Alpha", "#222222")
    color = node._compose_output_image(image, mask, "Color", "#112233")

    assert alpha.shape == (1, 1, 2, 4)
    np.testing.assert_allclose(alpha[0, 0, 1].numpy(), [0.0, 0.0, 0.0, 0.0])
    assert color.shape == (1, 1, 2, 3)
    np.testing.assert_allclose(color[0, 0, 1].numpy(), [17 / 255, 34 / 255, 51 / 255])


def test_apply_alpha_to_image():
    image_arr = np.zeros((4, 4, 3), dtype=np.uint8)
    image_arr[:] = (200, 100, 50)

    # Top-left 2x2 fully transparent, bottom-right 2x2 fully opaque.
    image = _make_image_tensor(image_arr)
    mask_arr = np.zeros((4, 4), dtype=np.float32)
    mask_arr[2:, 2:] = 1.0
    alpha_mask = torch.from_numpy(mask_arr).unsqueeze(0)

    node = PatternApplyAlphaToImage()
    (output,) = node.execute(image, alpha_mask, zero_transparent_rgb=True)

    assert output.dim() == 4
    assert output.shape == (1, 4, 4, 4)

    output_np = output[0].detach().cpu().numpy()
    expected_rgb = image_arr.astype(np.float32) / 255.0

    transparent = mask_arr == 0.0
    opaque = mask_arr == 1.0

    np.testing.assert_allclose(output_np[transparent, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(output_np[transparent, 3], 0.0, atol=1e-6)
    np.testing.assert_allclose(output_np[opaque, :3], expected_rgb[opaque], atol=1e-6)
    np.testing.assert_allclose(output_np[opaque, 3], 1.0, atol=1e-6)
