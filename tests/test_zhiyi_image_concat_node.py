"""Tests for `zhiyi_image_concat_node` module."""

import numpy as np
import torch

from src.Comfyui_Fd_Nodes.zhiyi_image_concat_node import ZhiYiImageConcatNode


def _make_image_tensor(height, width, color, channels=3):
    arr = np.full((height, width, channels), color, dtype=np.uint8)
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _pixel(tensor, y, x):
    return tuple((tensor[0, y, x].numpy() * 255).astype(np.uint8)[:3])


def test_horizontal_concat_pads_to_max_height_with_white():
    node = ZhiYiImageConcatNode()
    image_1 = _make_image_tensor(50, 100, RED)   # 矮宽
    image_2 = _make_image_tensor(120, 80, BLUE)  # 高窄

    (result,) = node.concat(image_1, image_2, "horizontal", "start", "white")

    assert result.shape == (1, 120, 180, 3)
    # image_1 内容完整保留（无截取）
    assert _pixel(result, 0, 0) == RED
    assert _pixel(result, 49, 99) == RED
    # image_2 内容完整保留
    assert _pixel(result, 0, 100) == BLUE
    assert _pixel(result, 119, 179) == BLUE
    # image_1 下方空缺处填白色
    assert _pixel(result, 119, 0) == WHITE
    assert _pixel(result, 50, 99) == WHITE


def test_vertical_concat_pads_to_max_width_with_white():
    node = ZhiYiImageConcatNode()
    image_1 = _make_image_tensor(50, 100, RED)
    image_2 = _make_image_tensor(80, 60, BLUE)

    (result,) = node.concat(image_1, image_2, "vertical", "start", "white")

    assert result.shape == (1, 130, 100, 3)
    assert _pixel(result, 0, 0) == RED
    assert _pixel(result, 49, 99) == RED
    assert _pixel(result, 50, 0) == BLUE
    assert _pixel(result, 129, 59) == BLUE
    # image_2 右侧空缺处填白色
    assert _pixel(result, 129, 99) == WHITE
    assert _pixel(result, 50, 99) == WHITE


def test_alignment_center_and_end_position_images():
    node = ZhiYiImageConcatNode()
    image_1 = _make_image_tensor(50, 100, RED)
    image_2 = _make_image_tensor(120, 80, BLUE)

    # center: image_1 上下各留 35px 空白
    (centered,) = node.concat(image_1, image_2, "horizontal", "center", "white")
    assert _pixel(centered, 0, 0) == WHITE
    assert _pixel(centered, 35, 0) == RED
    assert _pixel(centered, 84, 99) == RED
    assert _pixel(centered, 85, 99) == WHITE

    # end: image_1 贴底
    (ended,) = node.concat(image_1, image_2, "horizontal", "end", "white")
    assert _pixel(ended, 0, 0) == WHITE
    assert _pixel(ended, 70, 0) == RED
    assert _pixel(ended, 119, 99) == RED


def test_black_and_transparent_backgrounds():
    node = ZhiYiImageConcatNode()
    image_1 = _make_image_tensor(50, 100, RED)
    image_2 = _make_image_tensor(120, 80, BLUE)

    (black_result,) = node.concat(image_1, image_2, "horizontal", "start", "black")
    assert black_result.shape[-1] == 3
    assert _pixel(black_result, 119, 0) == BLACK

    (transparent_result,) = node.concat(image_1, image_2, "horizontal", "start", "transparent")
    assert transparent_result.shape[-1] == 4
    # 空白区 alpha 为 0，图像区 alpha 为 1（tensor 为 [0,1] 浮点）
    assert transparent_result[0, 119, 0, 3] == 0
    assert transparent_result[0, 0, 0, 3] == 1


def test_rgba_input_keeps_alpha_channel():
    node = ZhiYiImageConcatNode()
    image_1 = _make_image_tensor(50, 100, (255, 0, 0, 128), channels=4)
    image_2 = _make_image_tensor(50, 100, BLUE)

    (result,) = node.concat(image_1, image_2, "horizontal", "start", "white")

    assert result.shape == (1, 50, 200, 4)
    assert result[0, 0, 0, 3] == 1.0


def test_batch_inputs_pair_by_index():
    node = ZhiYiImageConcatNode()
    image_1 = torch.cat([_make_image_tensor(80, 100, RED), _make_image_tensor(80, 100, BLUE)], dim=0)
    image_2 = _make_image_tensor(80, 40, WHITE)

    (result,) = node.concat(image_1, image_2, "horizontal", "start", "black")

    # batch=2 的输入与 batch=1 的输入逐项配对，输出 batch=2
    assert result.shape == (2, 80, 140, 3)
    assert _pixel(result, 0, 0) == RED
    assert tuple((result[1, 0, 0].numpy() * 255).astype(np.uint8)) == BLUE
    assert _pixel(result, 0, 100) == WHITE


def test_concat_three_images_with_different_sizes():
    node = ZhiYiImageConcatNode()
    image_1 = _make_image_tensor(50, 100, RED)
    image_2 = _make_image_tensor(120, 80, BLUE)
    image_3 = _make_image_tensor(70, 60, (0, 255, 0))

    (result,) = node.concat(image_1, image_2, "horizontal", "start", "white", image_3=image_3)

    # 宽度为三张图之和，高度取最大值 120
    assert result.shape == (1, 120, 240, 3)
    assert _pixel(result, 0, 0) == RED
    assert _pixel(result, 0, 100) == BLUE
    assert _pixel(result, 0, 180) == (0, 255, 0)
    assert _pixel(result, 69, 239) == (0, 255, 0)
    # image_1/image_3 下方空缺处填白色
    assert _pixel(result, 119, 0) == WHITE
    assert _pixel(result, 119, 180) == WHITE


def test_concat_supports_gap_in_optional_inputs():
    node = ZhiYiImageConcatNode()
    image_1 = _make_image_tensor(50, 100, RED)
    image_3 = _make_image_tensor(50, 60, BLUE)

    # 只连 image_1 与 image_3（image_2 留空）同样可拼接
    (result,) = node.concat(image_1, None, "horizontal", "start", "white", image_3=image_3)

    assert result.shape == (1, 50, 160, 3)
    assert _pixel(result, 0, 0) == RED
    assert _pixel(result, 0, 100) == BLUE


def test_concat_requires_at_least_two_images():
    import pytest

    node = ZhiYiImageConcatNode()
    image_1 = _make_image_tensor(50, 100, RED)

    with pytest.raises(ValueError, match="至少需要连接两张图像"):
        node.concat(image_1=image_1, direction="horizontal", alignment="start", background="white")
