from collections.abc import Iterable

from ..config import CUSTOM_SERVICE_URL_PRESET


def service_url_preset_options(presets: Iterable[tuple[str, str]]) -> list[str]:
    preset_items = tuple(presets)
    names = [name for name, _url in preset_items]
    if any(not name or not url for name, url in preset_items):
        raise ValueError("服务 URL 预设的名称和地址不能为空")
    if len(names) != len(set(names)):
        raise ValueError("服务 URL 预设名称不能重复")
    return [CUSTOM_SERVICE_URL_PRESET, *names]


def resolve_service_url_preset(
    service_url: str,
    service_url_preset: str | None,
    presets: Iterable[tuple[str, str]],
) -> str:
    preset_items = tuple(presets)
    if not service_url_preset or service_url_preset == CUSTOM_SERVICE_URL_PRESET:
        return service_url

    preset_urls = dict(preset_items)
    if service_url_preset not in preset_urls:
        valid_options = ", ".join(service_url_preset_options(preset_items))
        raise RuntimeError(
            f"未知的 service_url_preset: {service_url_preset}；可选值为：{valid_options}"
        )
    return preset_urls[service_url_preset]
