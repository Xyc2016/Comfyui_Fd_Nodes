import os

FD_OSS_ACCESS_KEY_ID = os.getenv("FD_OSS_ACCESS_KEY_ID")
FD_OSS_ACCESS_KEY_SECRET = os.getenv("FD_OSS_ACCESS_KEY_SECRET")
FD_OSS_BUCKET_NAME = os.getenv("FD_OSS_BUCKET_NAME")
FD_OSS_ENDPOINT = os.getenv("FD_OSS_ENDPOINT")
FD_OSS_URL_PREFIX = os.getenv("FD_OSS_URL_PREFIX")
FD_OSS_URL_PATH_PREFIX = os.getenv("FD_OSS_URL_PATH_PREFIX", "devops/comfyui/text_img")
FD_OSS_URL_PATH_PREFIX_GEMINI =  os.getenv("FD_OSS_URL_PATH_PREFIX_GEMINI", "devops/comfyui/segment_img")
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

FD_OSS_URL_PATH_PREFIX_BEFORE_GEN = os.getenv("FD_OSS_URL_PATH_PREFIX_BEFORE_GEN", "devops/comfyui/segment_img")
FD_AISTUDIO_PUBLISH_URL = os.getenv("FD_AISTUDIO_PUBLISH_URL", "http://121.40.67.98:2003/api/tasks/publish")
FD_REMOVE_BG_BY_MEITU_URL = os.getenv(
    "FD_REMOVE_BG_BY_MEITU_URL",
    "http://image-server-internal.zhiyi.com.cn/api-server-gray/detail-image/image/remove_bg_by_meitu",
)
FD_OSS_URL_PATH_PREFIX_REMOVE_BG = os.getenv("FD_OSS_URL_PATH_PREFIX_REMOVE_BG", "devops/comfyui/remove_bg")
FD_SAM2_SEGMENT_URL = os.getenv("FD_SAM2_SEGMENT_URL", "http://model-api-sam2-hiera-base-plus-svc.online-server-gray:8000/v1/segment")
FD_DWPOSE_POSE_URL = os.getenv("FD_DWPOSE_POSE_URL", "http://model-api-dwpose-svc.online-server-gray:8001/v1/pose")

assert FD_LITELLM_BASE_URL, "FD_LITELLM_BASE_URL is not set"
assert FD_LITELLM_API_KEY, "FD_LITELLM_API_KEY is not set"
