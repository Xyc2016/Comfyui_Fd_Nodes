import pytest

from src.Comfyui_Fd_Nodes.utils.gpt_image_size import (
    _SIZE_OVERRIDE_RE,
    resolution_to_edit_size,
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


def test_size_override_regex_matches_positive_integer_dimensions():
    assert _SIZE_OVERRIDE_RE.match("1537x1025")
    assert _SIZE_OVERRIDE_RE.match("2048x2048")
    assert _SIZE_OVERRIDE_RE.match("1x1")
    assert not _SIZE_OVERRIDE_RE.match("0x1024")
    assert not _SIZE_OVERRIDE_RE.match("1536X1024")
