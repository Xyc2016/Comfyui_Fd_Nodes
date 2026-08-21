import os
from urllib.parse import urlparse, urlunparse

FD_OSS_ACCESS_KEY_ID = os.getenv("FD_OSS_ACCESS_KEY_ID")
FD_OSS_ACCESS_KEY_SECRET = os.getenv("FD_OSS_ACCESS_KEY_SECRET")
FD_OSS_BUCKET_NAME = os.getenv("FD_OSS_BUCKET_NAME")
FD_OSS_ENDPOINT = os.getenv("FD_OSS_ENDPOINT")
FD_OSS_URL_PREFIX = os.getenv("FD_OSS_URL_PREFIX")
FD_OSS_STS_URL = os.getenv("FD_OSS_STS_URL")
FD_OSS_STS_KEY = os.getenv("FD_OSS_STS_KEY")
FD_OSS_STS_TIMEOUT = os.getenv("FD_OSS_STS_TIMEOUT", "10")
FD_OSS_URL_PATH_PREFIX = os.getenv("FD_OSS_URL_PATH_PREFIX", "devops/comfyui/text_img")
FD_OSS_URL_PATH_PREFIX_GEMINI =  os.getenv("FD_OSS_URL_PATH_PREFIX_GEMINI", "devops/comfyui/segment_img")
FD_OSS_URL_PATH_PREFIX_GPT_IMAGE = os.getenv("FD_OSS_URL_PATH_PREFIX_GPT_IMAGE", FD_OSS_URL_PATH_PREFIX_GEMINI)
FD_GEMINI_URL = os.getenv("FD_GEMINI_URL")
FD_DOUBAO_KEY = os.getenv("FD_DOUBAO_KEY")
FD_DOUBAO_URL = os.getenv("FD_DOUBAO_URL")
FD_GEMINI_WEBHOOK_URL = os.getenv("FD_GEMINI_WEBHOOK_URL")
FD_FLUX2KLEIN_URL = os.getenv("FD_FLUX2KLEIN_URL")
FD_FLUX2KLEIN_USERNAME = os.getenv("FD_FLUX2KLEIN_USERNAME")
FD_FLUX2KLEIN_PASSWORD = os.getenv("FD_FLUX2KLEIN_PASSWORD")
FD_Z_IMAGE_TURBO_URL = os.getenv("FD_Z_IMAGE_TURBO_URL")
FD_Z_IMAGE_TURBO_USERNAME = os.getenv("FD_Z_IMAGE_TURBO_USERNAME")
FD_Z_IMAGE_TURBO_PASSWORD = os.getenv("FD_Z_IMAGE_TURBO_PASSWORD")
FD_OSS_URL_PATH_PREFIX_FLUX =  os.getenv("FD_OSS_URL_PATH_PREFIX_FLUX", "devops/comfyui/segment_img")
FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL = os.getenv("FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL")

FD_LITELLM_BASE_URL = os.getenv("FD_LITELLM_BASE_URL")
FD_LITELLM_API_KEY = os.getenv("FD_LITELLM_API_KEY")
FD_GPT_IMAGE_BACKEND = os.getenv("FD_GPT_IMAGE_BACKEND", "image_generation")


def _derive_gpt_image_edit_url() -> str:
    source_url = (FD_GEMINI_URL or "").strip()
    if not source_url:
        return "http://image-generation.online-server-gray.svc.cluster.local/image/edit"
    if "://" not in source_url:
        source_url = f"http://{source_url}"

    parsed = urlparse(source_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if not path:
        path = "/image/edit"
    elif path.endswith("/image/gemini_image"):
        path = f"{path[:-len('/gemini_image')]}/edit"
    elif path.endswith("/gemini_image"):
        path = f"{path[:-len('/gemini_image')]}/image/edit"
    elif path.endswith("/image"):
        path = f"{path}/edit"
    else:
        path = f"{path}/image/edit"

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


FD_GPT_IMAGE_EDIT_URL = os.getenv("FD_GPT_IMAGE_EDIT_URL") or _derive_gpt_image_edit_url()

FD_SEEDREAM_BACKEND = os.getenv("FD_SEEDREAM_BACKEND", "image_generation")


def _derive_image_generate_url() -> str:
    source_url = (FD_GEMINI_URL or "").strip() or FD_GPT_IMAGE_EDIT_URL
    if "://" not in source_url:
        source_url = f"http://{source_url}"

    parsed = urlparse(source_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if not path:
        path = "/image/generate"
    elif path.endswith("/image/gemini_image"):
        path = f"{path[:-len('/gemini_image')]}/generate"
    elif path.endswith("/gemini_image"):
        path = f"{path[:-len('/gemini_image')]}/image/generate"
    elif path.endswith("/image/edit"):
        path = f"{path[:-len('/edit')]}/generate"
    elif path.endswith("/image"):
        path = f"{path}/generate"
    else:
        path = f"{path}/image/generate"

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


FD_IMAGE_GENERATE_URL = os.getenv("FD_IMAGE_GENERATE_URL") or _derive_image_generate_url()

FD_OSS_URL_PATH_PREFIX_BEFORE_GEN = os.getenv("FD_OSS_URL_PATH_PREFIX_BEFORE_GEN", "devops/comfyui/segment_img")
FD_AISTUDIO_PUBLISH_URL = os.getenv("FD_AISTUDIO_PUBLISH_URL", "http://121.40.67.98:2003/api/tasks/publish")
FD_REMOVE_BG_BY_MEITU_URL = os.getenv(
    "FD_REMOVE_BG_BY_MEITU_URL",
    "http://image-server-internal.zhiyi.com.cn/api-server-gray/detail-image/image/remove_bg_by_meitu",
)
FD_OSS_URL_PATH_PREFIX_REMOVE_BG = os.getenv("FD_OSS_URL_PATH_PREFIX_REMOVE_BG", "devops/comfyui/remove_bg")
FD_SAM2_SEGMENT_URL = os.getenv("FD_SAM2_SEGMENT_URL", "http://model-api-sam2-hiera-base-plus-svc.online-server-gray:8000/v1/segment")
DEFAULT_DWPOSE_POSE_URL = "http://model-api-dwpose-svc.online-server-gray:8001/v1/pose"
FD_DWPOSE_POSE_URL = os.getenv("FD_DWPOSE_POSE_URL") or DEFAULT_DWPOSE_POSE_URL


def _derive_dwpose_preprocess_url(endpoint: str) -> str:
    source_url = (FD_DWPOSE_POSE_URL or DEFAULT_DWPOSE_POSE_URL).strip()
    if "://" not in source_url:
        source_url = f"http://{source_url}"

    parsed = urlparse(source_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if not path:
        path = f"/v1/{endpoint}"
    elif path.endswith("/v1/pose"):
        path = f"{path[:-len('/pose')]}/{endpoint}"
    elif path.endswith("/pose"):
        base_path = path[:-len("/pose")].rstrip("/")
        path = f"{base_path}/v1/{endpoint}" if base_path else f"/v1/{endpoint}"
    elif path.endswith("/v1"):
        path = f"{path}/{endpoint}"
    else:
        path = f"{path}/v1/{endpoint}"

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


FD_CONTROLNET_AUX_LINEART_URL = os.getenv("FD_CONTROLNET_AUX_LINEART_URL") or _derive_dwpose_preprocess_url("lineart")
FD_CONTROLNET_AUX_DEPTH_ANYTHING_V2_URL = os.getenv("FD_CONTROLNET_AUX_DEPTH_ANYTHING_V2_URL") or _derive_dwpose_preprocess_url("depth-anything-v2")
FD_RMBG_URL = os.getenv("FD_RMBG_URL", "http://10.1.0.230:8003/v1/rmbg")
FD_CLOTHES_SEGMENT_URL = os.getenv("FD_CLOTHES_SEGMENT_URL", "http://10.1.0.230:8003/v1/segment/clothes")
FD_FASHION_SEGMENT_URL = os.getenv("FD_FASHION_SEGMENT_URL", "http://10.1.0.230:8003/v1/segment/fashion")
FD_BODY_SEGMENT_URL = os.getenv("FD_BODY_SEGMENT_URL", "http://10.1.0.230:8003/v1/segment/body")

CUSTOM_SERVICE_URL_PRESET = "自定义（使用 service_url）"
DWPOSE_SERVICE_URL_PRESETS = (
    ("K8s 灰度", "http://model-api-dwpose-svc.online-server-gray:8001"),
    ("10.1.0.230", "http://10.1.0.230:8003"),
)
RMBG_SERVICE_URL_PRESETS = (
    ("10.1.0.230", "http://10.1.0.230:8003"),
)

DWPOSE_SERVICE_ENVS = {
    "gray": "http://model-api-dwpose-svc.online-server-gray:8001",
    "dev": "http://10.1.0.230:8003",
}
RMBG_SERVICE_ENVS = {
    "gray": "http://model-api-dwpose-svc.online-server-gray:8001",
    "dev": "http://10.1.0.230:8003",
}
SAM2_SERVICE_ENVS = {
    "gray": "http://model-api-sam2-hiera-base-plus-svc.online-server-gray:8000",
    "dev": "http://10.1.0.230:8002",
}

assert FD_LITELLM_BASE_URL, "FD_LITELLM_BASE_URL is not set"
assert FD_LITELLM_API_KEY, "FD_LITELLM_API_KEY is not set"
