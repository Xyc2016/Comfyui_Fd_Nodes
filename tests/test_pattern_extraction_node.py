"""Tests for `pattern_extraction_node` module."""

import json

import numpy as np
import torch

from src.Comfyui_Fd_Nodes import nodes
from src.Comfyui_Fd_Nodes.pattern_extraction_node import (
    PatternChooseBackgroundPair,
    PatternDualBackgroundToRGBA,
)


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

    rgba_image, alpha_mask, meta_json = node.execute(
        image_a, image_b, sample_span=24, alpha_floor=0.02
    )

    assert rgba_image.dim() == 4
    assert rgba_image.shape[0] == 1
    assert rgba_image.shape[-1] == 4
    assert alpha_mask.dim() == 3
    assert alpha_mask.shape[0] == 1

    meta = json.loads(meta_json)
    assert isinstance(meta, list) and len(meta) == 1
    assert "bg_a_rgb" in meta[0]
    assert "bg_b_rgb" in meta[0]


def test_nodes_registered_in_mappings():
    assert "PatternChooseBackgroundPair+" in nodes.NODE_CLASS_MAPPINGS
    assert "PatternDualBackgroundToRGBA+" in nodes.NODE_CLASS_MAPPINGS
    assert nodes.NODE_CLASS_MAPPINGS["PatternChooseBackgroundPair+"] is PatternChooseBackgroundPair
    assert nodes.NODE_CLASS_MAPPINGS["PatternDualBackgroundToRGBA+"] is PatternDualBackgroundToRGBA

    assert "PatternChooseBackgroundPair+" in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert "PatternDualBackgroundToRGBA+" in nodes.NODE_DISPLAY_NAME_MAPPINGS
