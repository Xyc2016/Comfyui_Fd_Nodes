import logging

import pytest

from src.Comfyui_Fd_Nodes.utils.seedream_image_size import (
    SEEDREAM_ASPECT_RATIOS,
    SEEDREAM_IMAGE_SIZES,
    SEEDREAM_SIZE_MAP,
    resolution_to_seedream_size,
)


@pytest.mark.parametrize(
    ("resolution", "aspect_ratio", "expected"),
    [
        ("1K", "1:1", "1024x1024"),
        ("1K", "2:3", "832x1248"),
        ("1K", "3:2", "1248x832"),
        ("1K", "3:4", "864x1152"),
        ("1K", "4:3", "1152x864"),
        ("1K", "4:5", "896x1120"),
        ("1K", "5:4", "1120x896"),
        ("1K", "16:9", "1280x720"),
        ("1K", "9:16", "720x1280"),
        ("1K", "21:9", "1536x640"),
        ("1K", "9:21", "640x1536"),
        ("2K", "1:1", "2048x2048"),
        ("2K", "2:3", "1664x2496"),
        ("2K", "3:2", "2496x1664"),
        ("2K", "3:4", "1536x2048"),
        ("2K", "4:3", "2048x1536"),
        ("2K", "4:5", "1792x2240"),
        ("2K", "5:4", "2240x1792"),
        ("2K", "16:9", "2560x1440"),
        ("2K", "9:16", "1440x2560"),
        ("2K", "21:9", "3072x1280"),
        ("2K", "9:21", "1280x3072"),
        ("3K", "4:5", "2688x3360"),
        ("3K", "5:4", "3360x2688"),
        ("4K", "1:1", "2880x2880"),
        ("4K", "4:5", "2304x2880"),
        ("4K", "5:4", "2880x2304"),
        ("4K", "16:9", "3840x2160"),
        ("4K", "9:21", "1648x3840"),
    ],
)
def test_seedream_size_maps_resolution_and_aspect_ratio_to_pixels(resolution, aspect_ratio, expected):
    assert resolution_to_seedream_size(resolution, aspect_ratio) == expected


def test_seedream_size_defaults_to_square_ratio():
    assert resolution_to_seedream_size("2K") == "2048x2048"
    assert resolution_to_seedream_size("4K", "") == "2880x2880"
    assert resolution_to_seedream_size("4K", "unknown") == "2880x2880"


def test_seedream_size_warns_on_unknown_ratio(caplog):
    with caplog.at_level(logging.WARNING, logger="src.Comfyui_Fd_Nodes.utils.seedream_image_size"):
        resolution_to_seedream_size("2K", "3:7")
    assert any("回退 1:1" in record.message for record in caplog.records)


@pytest.mark.parametrize("resolution", ["unknown", "", "5K"])
def test_seedream_size_rejects_unknown_resolution(resolution):
    with pytest.raises(
        ValueError,
        match=rf"Invalid Seedream resolution {resolution!r}; expected one of \['4K', '3K', '2K', '1K'\]",
    ):
        resolution_to_seedream_size(resolution, "21:9")


def test_seedream_size_options_expose_1k():
    assert SEEDREAM_IMAGE_SIZES == ["4K", "3K", "2K", "1K"]
    assert SEEDREAM_ASPECT_RATIOS == ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9", "9:21"]


def test_seedream_size_map_covers_every_exposed_ratio():
    """枚举暴露给 UI 的每个比例都必须能查到像素，防止枚举与映射表再次脱节。"""
    for resolution in SEEDREAM_IMAGE_SIZES:
        assert set(SEEDREAM_ASPECT_RATIOS) <= set(SEEDREAM_SIZE_MAP[resolution]), (
            f"{resolution} 档映射表缺少枚举中的比例: {set(SEEDREAM_ASPECT_RATIOS) - set(SEEDREAM_SIZE_MAP[resolution])}"
        )


@pytest.mark.parametrize(
    "resolution",
    ["1K", "2K", "3K", "4K"],
)
def test_seedream_size_map_ratios_match_expected_value(resolution):
    """每个比例的宽高比值必须接近标注比例，防止映射表填错方向。"""
    for ratio in SEEDREAM_SIZE_MAP[resolution]:
        w_str, h_str = SEEDREAM_SIZE_MAP[resolution][ratio].split("x")
        w, h = int(w_str), int(h_str)
        expected_w, expected_h = (int(ratio.split(":")[0]), int(ratio.split(":")[1]))
        # 允许 4% 的取整误差
        assert abs(w / h - expected_w / expected_h) < 0.04 * expected_w / expected_h, (
            f"{resolution}/{ratio} 映射为 {w}x{h}，偏差过大"
        )
