"""V2 节点环境选择：灰度/开发/自定义。

env 下拉使用纯中文字符串（单选 Combo 不支持 {text, value} 对象，选中后会被序列化成对象），
后端在 resolve_service_env 中把中文文本归一化为 gray/dev/custom key。
"""

from ..config import CUSTOM_SERVICE_URL_PRESET

CUSTOM_ENV = "custom"

ENV_OPTIONS = [
    "灰度环境",
    "开发环境",
    CUSTOM_SERVICE_URL_PRESET,
]

# 前端提交的中文文本 -> 环境 key
ENV_TEXT_TO_VALUE = {
    "灰度环境": "gray",
    "开发环境": "dev",
    CUSTOM_SERVICE_URL_PRESET: CUSTOM_ENV,
}
ENV_VALUE_TO_TEXT = {value: text for text, value in ENV_TEXT_TO_VALUE.items()}


def service_env_options() -> list[str]:
    return ENV_OPTIONS


def normalize_env(env: str | dict | None) -> str | None:
    """归一化为 gray/dev/custom/None；兼容旧工作流中保存的 {text, value} 对象。"""
    if env is None or env == "":
        return None
    if isinstance(env, dict):
        env = env.get("value") or env.get("text")
    if not isinstance(env, str):
        raise RuntimeError(f"未知的 env: {env}；可选值为：{', '.join(ENV_OPTIONS)}")
    if env in ENV_TEXT_TO_VALUE:
        return ENV_TEXT_TO_VALUE[env]
    if env in ENV_VALUE_TO_TEXT:
        return env
    raise RuntimeError(f"未知的 env: {env}；可选值为：{', '.join(ENV_OPTIONS)}")


def resolve_service_env(env: str | dict | None, service_envs: dict[str, str]) -> str | None:
    """gray/dev 命中环境表返回 base url；custom/None/空串返回 None；其他值抛 RuntimeError。"""
    normalized = normalize_env(env)
    if normalized is None or normalized == CUSTOM_ENV:
        return None
    if normalized not in service_envs:
        valid_options = ", ".join([*service_envs, CUSTOM_ENV])
        raise RuntimeError(
            f"未知的 env: {normalized}；可选值为：{valid_options}"
        )
    return service_envs[normalized]
