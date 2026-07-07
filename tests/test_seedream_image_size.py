from src.Comfyui_Fd_Nodes.utils.seedream_image_size import (
    SEEDREAM_ASPECT_RATIOS,
    SEEDREAM_IMAGE_SIZES,
    resolution_to_seedream_size,
)


def test_seedream_size_maps_resolution_and_aspect_ratio_to_pixels():
    assert resolution_to_seedream_size("2K", "1:1") == "2048x2048"
    assert resolution_to_seedream_size("2K", "3:4") == "1728x2304"
    assert resolution_to_seedream_size("3K", "21:9") == "4704x2016"
    assert resolution_to_seedream_size("4K", "9:16") == "3040x5504"


def test_seedream_size_defaults_to_square_ratio():
    assert resolution_to_seedream_size("2K") == "2048x2048"
    assert resolution_to_seedream_size("4K", "") == "4096x4096"
    assert resolution_to_seedream_size("4K", "unknown") == "4096x4096"


def test_seedream_size_falls_back_to_2k_for_unknown_resolution():
    assert resolution_to_seedream_size("1K", "3:4") == "1728x2304"
    assert resolution_to_seedream_size("unknown", "21:9") == "3136x1344"


def test_seedream_size_options_do_not_expose_1k():
    assert SEEDREAM_IMAGE_SIZES == ["4K", "3K", "2K"]
    assert "1K" not in SEEDREAM_IMAGE_SIZES
    assert SEEDREAM_ASPECT_RATIOS == ["1:1", "3:4", "4:3", "16:9", "9:16", "3:2", "2:3", "21:9"]
