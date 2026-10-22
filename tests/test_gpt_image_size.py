import pytest

from src.Comfyui_Fd_Nodes.utils.gpt_image_size import (
    _SIZE_OVERRIDE_RE,
    resolution_to_edit_size,
    resolution_to_image_generation_edit_size,
    resolve_gpt_image_size,
)


def test_resolve_gpt_image_size_empty_override_falls_back_to_preset():
    """size_override 为空时原样返回 preset 和比例"""
    size, ratio = resolve_gpt_image_size(
        preset_size="2K", aspect_ratio="3:4", size_override=""
    )
    assert size == "2K"
    assert ratio == "3:4"


def test_resolve_gpt_image_size_none_override_falls_back_to_preset():
    """size_override=None 等价空字符串"""
    size, ratio = resolve_gpt_image_size(
        preset_size="2K", aspect_ratio="3:4", size_override=None
    )
    assert size == "2K"
    assert ratio == "3:4"


def test_resolve_gpt_image_size_whitespace_only_override_falls_back():
    """纯空白视为空"""
    size, ratio = resolve_gpt_image_size(
        preset_size="2K", aspect_ratio="3:4", size_override="   "
    )
    assert size == "2K"
    assert ratio == "3:4"


def test_resolve_gpt_image_size_valid_override_is_passed_through():
    """合法 WxH 原样返回并清空比例"""
    size, ratio = resolve_gpt_image_size(
        preset_size="2K", aspect_ratio="3:4", size_override="1537x1025"
    )
    assert size == "1537x1025"
    assert ratio == ""


def test_resolve_gpt_image_size_odd_dimensions_are_accepted():
    """非 16 对齐的奇怪尺寸也接受，原样透传"""
    size, ratio = resolve_gpt_image_size(
        preset_size="2K", aspect_ratio="9:16", size_override="17x19"
    )
    assert size == "17x19"
    assert ratio == ""


@pytest.mark.parametrize(
    "bad_value",
    [
        "0x1024",
        "1536X1024",
        "1536 x 1024",
        " 1536x1024",
        "1536x1024 ",
        "1024x",
        "x1024",
        "1K",
        "1024.5x768",
        "-1x1024",
    ],
)
def test_resolve_gpt_image_size_invalid_override_raises_value_error(bad_value):
    """非 `正整数x正整数` 一律 ValueError"""
    with pytest.raises(ValueError, match="size_override 需为 WIDTHxHEIGHT 格式"):
        resolve_gpt_image_size(
            preset_size="2K", aspect_ratio="", size_override=bad_value
        )


def test_resolve_gpt_image_size_ratio_is_preserved_when_no_override():
    """不填覆盖时比例不受影响"""
    size, ratio = resolve_gpt_image_size(
        preset_size="4K", aspect_ratio="9:16", size_override=""
    )
    assert size == "4K"
    assert ratio == "9:16"


def test_old_preset_maps_are_not_affected():
    """旧 resolution_to_edit_size 映射不变"""
    assert resolution_to_edit_size("4K", "9:16") == "2160x3840"
    assert resolution_to_edit_size("2K", "") == "2048x2048"
    assert resolution_to_edit_size("1K", "21:9") == "1456x624"


def test_image_generation_edit_size_4k_21x9_keeps_ratio():
    """4K+21:9 必须输出 21:9 像素（回归：曾误用 16:9 的 3840x2160，网关透传显式像素导致出图 16:9）"""
    assert resolution_to_image_generation_edit_size("4K", "21:9") == "3840x1648"


@pytest.mark.parametrize("resolution", ["1K", "2K", "4K"])
@pytest.mark.parametrize(
    "aspect_ratio",
    ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9", "9:21"],
)
def test_image_generation_edit_size_output_is_tier_or_ratio_exact_pixels(resolution, aspect_ratio):
    """节点发给网关的 size 只允许两种形态：档位简写（网关查表换算），或比例精确
    （偏差<0.5%、16 倍数）的像素串——显式像素网关会原样透传，比例错了没人兜底"""
    out = resolution_to_image_generation_edit_size(resolution, aspect_ratio)
    if out in ("1K", "2K", "4K"):
        return
    width, height = (int(part) for part in out.split("x"))
    ratio_w, ratio_h = (int(part) for part in aspect_ratio.split(":"))
    assert abs(width / height - ratio_w / ratio_h) / (ratio_w / ratio_h) < 0.005
    assert width % 16 == 0 and height % 16 == 0


@pytest.mark.parametrize(
    "resolution,aspect_ratio,expected",
    [
        ("4K", "16:9", "3840x2160"),
        ("4K", "9:16", "2160x3840"),
        ("4K", "9:21", "4K"),
        ("4K", "3:4", "4K"),
        ("4K", "", "4K"),
        ("2K", "21:9", "2K"),
        ("1K", "16:9", "1K"),
        ("1080P", "16:9", "2K"),
    ],
)
def test_image_generation_edit_size_tiers(resolution, aspect_ratio, expected):
    """其余档位/比例行为保持不变：非特判组合原样返回档位简写，未知档位回退 2K"""
    assert resolution_to_image_generation_edit_size(resolution, aspect_ratio) == expected


def test_size_override_regex_matches_positive_integer_dimensions():
    assert _SIZE_OVERRIDE_RE.match("1537x1025")
    assert _SIZE_OVERRIDE_RE.match("2048x2048")
    assert _SIZE_OVERRIDE_RE.match("1x1")
    assert not _SIZE_OVERRIDE_RE.match("0x1024")
    assert not _SIZE_OVERRIDE_RE.match("1536X1024")
