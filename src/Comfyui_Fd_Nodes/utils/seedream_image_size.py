import logging

from .logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)

SEEDREAM_IMAGE_SIZES = ["4K", "3K", "2K", "1K"]
SEEDREAM_ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9", "9:21"]

# 与 model-servings 服务端 ASPECT_RATIO_SIZE_TABLE（Seedream 通用档）保持一致；
# 3K 档上游不存在，resolve 阶段按 2K 请求。
SEEDREAM_SIZE_MAP = {
    "1K": {
        "1:1": "1024x1024",
        "2:3": "832x1248",
        "3:2": "1248x832",
        "3:4": "864x1152",
        "4:3": "1152x864",
        "4:5": "896x1120",
        "5:4": "1120x896",
        "16:9": "1280x720",
        "9:16": "720x1280",
        "21:9": "1536x640",
        "9:21": "640x1536",
    },
    "2K": {
        "1:1": "2048x2048",
        "2:3": "1664x2496",
        "3:2": "2496x1664",
        "3:4": "1536x2048",
        "4:3": "2048x1536",
        "4:5": "1792x2240",
        "5:4": "2240x1792",
        "16:9": "2560x1440",
        "9:16": "1440x2560",
        "21:9": "3072x1280",
        "9:21": "1280x3072",
    },
    "3K": {
        "1:1": "3072x3072",
        "2:3": "2496x3744",
        "3:2": "3744x2496",
        "3:4": "2304x3072",
        "4:3": "3072x2304",
        "4:5": "2688x3360",
        "5:4": "3360x2688",
        "16:9": "3840x2160",
        "9:16": "2160x3840",
        "21:9": "4608x1920",
        "9:21": "1920x4608",
    },
    "4K": {
        "1:1": "2880x2880",
        "2:3": "1920x2880",
        "3:2": "2880x1920",
        "3:4": "2160x2880",
        "4:3": "2880x2160",
        "4:5": "2304x2880",
        "5:4": "2880x2304",
        "16:9": "3840x2160",
        "9:16": "2160x3840",
        "21:9": "3840x1648",
        "9:21": "1648x3840",
    },
}


def resolution_to_seedream_size(resolution: str, aspect_ratio: str = "1:1") -> str:
    if resolution not in SEEDREAM_SIZE_MAP:
        raise ValueError(f"Invalid Seedream resolution {resolution!r}; expected one of {SEEDREAM_IMAGE_SIZES}")
    if aspect_ratio not in SEEDREAM_SIZE_MAP[resolution]:
        logger.warning(
            "Seedream ratio %r 不在 %s 档映射表中，回退 1:1",
            aspect_ratio,
            resolution,
        )
        aspect_ratio = "1:1"
    return SEEDREAM_SIZE_MAP[resolution][aspect_ratio]
