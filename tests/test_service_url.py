import pytest

from src.Comfyui_Fd_Nodes.config import CUSTOM_SERVICE_URL_PRESET
from src.Comfyui_Fd_Nodes.utils.service_url import (
    resolve_service_url_preset,
    service_url_preset_options,
)


PRESETS = (
    ("部署 A", "http://service-a:8001"),
    ("部署 B", "https://service-b"),
)


def test_service_url_preset_options_puts_custom_first():
    assert service_url_preset_options(PRESETS) == [
        CUSTOM_SERVICE_URL_PRESET,
        "部署 A",
        "部署 B",
    ]


def test_resolve_service_url_preset_preserves_custom_and_legacy_values():
    assert resolve_service_url_preset("custom:9000", None, PRESETS) == "custom:9000"
    assert resolve_service_url_preset("custom:9000", "", PRESETS) == "custom:9000"
    assert (
        resolve_service_url_preset("custom:9000", CUSTOM_SERVICE_URL_PRESET, PRESETS)
        == "custom:9000"
    )


def test_resolve_service_url_preset_named_deployment_overrides_custom_url():
    assert (
        resolve_service_url_preset("custom:9000", "部署 A", PRESETS)
        == "http://service-a:8001"
    )


def test_resolve_service_url_preset_rejects_unknown_deployment():
    with pytest.raises(RuntimeError, match="未知的 service_url_preset"):
        resolve_service_url_preset("custom:9000", "不存在", PRESETS)


def test_service_url_preset_options_rejects_invalid_definitions():
    with pytest.raises(ValueError, match="不能为空"):
        service_url_preset_options((("", "http://service"),))
    with pytest.raises(ValueError, match="不能为空"):
        service_url_preset_options((("部署", ""),))
    with pytest.raises(ValueError, match="不能重复"):
        service_url_preset_options((("部署", "http://a"), ("部署", "http://b")))
