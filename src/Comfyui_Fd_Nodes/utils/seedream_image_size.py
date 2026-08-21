SEEDREAM_IMAGE_SIZES = ["4K", "3K", "2K", "1K"]
SEEDREAM_ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9", "9:21"]

SEEDREAM_SIZE_MAP = {
    "1K": {
        "1:1": "1024x1024",
        "4:3": "1152x864",
        "3:4": "864x1152",
        "16:9": "1424x800",
        "9:16": "800x1424",
        "3:2": "1248x832",
        "2:3": "832x1248",
        "21:9": "1568x672",
        "9:21": "640x1536",
    },
    "2K": {
        "1:1": "2048x2048",
        "3:4": "1728x2304",
        "4:3": "2304x1728",
        "16:9": "2848x1600",
        "9:16": "1600x2848",
        "3:2": "2496x1664",
        "2:3": "1664x2496",
        "21:9": "3136x1344",
        "9:21": "1280x3072",
    },
    "3K": {
        "1:1": "3072x3072",
        "3:4": "2592x3456",
        "4:3": "3456x2592",
        "16:9": "4096x2304",
        "9:16": "2304x4096",
        "2:3": "2496x3744",
        "3:2": "3744x2496",
        "21:9": "4704x2016",
        "9:21": "1280x3072",
    },
    "4K": {
        "1:1": "4096x4096",
        "3:4": "3520x4704",
        "4:3": "4704x3520",
        "16:9": "5504x3040",
        "9:16": "3040x5504",
        "2:3": "3328x4992",
        "3:2": "4992x3328",
        "21:9": "6240x2656",
        "9:21": "1648x3840",
    },
}


def resolution_to_seedream_size(resolution: str, aspect_ratio: str = "1:1") -> str:
    if resolution not in SEEDREAM_SIZE_MAP:
        raise ValueError(f"Invalid Seedream resolution {resolution!r}; expected one of {SEEDREAM_IMAGE_SIZES}")
    normalized_aspect_ratio = aspect_ratio if aspect_ratio in SEEDREAM_SIZE_MAP[resolution] else "1:1"
    return SEEDREAM_SIZE_MAP[resolution][normalized_aspect_ratio]
