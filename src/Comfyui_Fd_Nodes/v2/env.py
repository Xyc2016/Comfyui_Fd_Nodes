"""V2 节点环境选择：灰度/开发/自定义。"""

from ..config import CUSTOM_SERVICE_URL_PRESET

CUSTOM_ENV = "custom"

ENV_OPTIONS = [
    {"text": "灰度环境", "value": "gray"},
    {"text": "开发环境", "value": "dev"},
    {"text": f"{CUSTOM_SERVICE_URL_PRESET}", "value": CUSTOM_ENV},
]


def service_env_options() -> list[dict[str, str]]:
    return ENV_OPTIONS


def resolve_service_env(env: str | None, service_envs: dict[str, str]) -> str | None:
    """gray/dev 命中环境表返回 base url；custom/None/空串原样返回；其他值抛 RuntimeError。"""
    if env is None or env == "" or env == CUSTOM_ENV:
        return None
    if not isinstance(env, str) or env not in service_envs:
        valid_options = ", ".join([*service_envs, CUSTOM_ENV])
        raise RuntimeError(
            f"未知的 env: {env}；可选值为：{valid_options}"
        )
    return service_envs[env]
