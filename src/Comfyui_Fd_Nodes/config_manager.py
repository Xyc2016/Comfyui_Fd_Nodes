from .config import (
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
)

_DEFAULT = {"base_url": FD_LITELLM_BASE_URL, "api_key": FD_LITELLM_API_KEY}

def load_config():
    return dict(_DEFAULT)


def save_config(base_url: str, api_key: str):
    return
