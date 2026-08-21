import pytest

from src.Comfyui_Fd_Nodes.utils.seedream_image_size import (
    SEEDREAM_ASPECT_RATIOS,
    SEEDREAM_IMAGE_SIZES,
    resolution_to_seedream_size,
)


@pytest.mark.parametrize(
    ("resolution", "aspect_ratio", "expected"),
    [
        ("1K", "1:1", "1024x1024"),
        ("1K", "4:3", "1152x864"),
        ("1K", "3:4", "864x1152"),
        ("1K", "16:9", "1424x800"),
        ("1K", "9:16", "800x1424"),
        ("1K", "3:2", "1248x832"),
        ("1K", "2:3", "832x1248"),
        ("1K", "21:9", "1568x672"),
        ("2K", "1:1", "2048x2048"),
        ("2K", "3:4", "1728x2304"),
        ("3K", "21:9", "4704x2016"),
        ("2K", "9:21", "1280x3072"),
        ("4K", "9:16", "3040x5504"),
        ("4K", "9:21", "1648x3840"),
    ],
)
def test_seedream_size_maps_resolution_and_aspect_ratio_to_pixels(resolution, aspect_ratio, expected):
    assert resolution_to_seedream_size(resolution, aspect_ratio) == expected


def test_seedream_size_defaults_to_square_ratio():
    assert resolution_to_seedream_size("2K") == "2048x2048"
    assert resolution_to_seedream_size("4K", "") == "4096x4096"
    assert resolution_to_seedream_size("4K", "unknown") == "4096x4096"


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
