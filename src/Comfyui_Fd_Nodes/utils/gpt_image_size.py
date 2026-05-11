def resolution_to_edit_size(resolution: str, aspect_ratio: str) -> str:
    # gpt-image-2 accepts flexible sizes, but they must satisfy strict bounds:
    # edge <= 3840, edges are multiples of 16, ratio <= 3:1,
    # and total pixels stay within [655360, 8294400].
    # Use fixed valid presets here so the Comfy node never emits illegal sizes.
    size_map = {
        "1K": {
            "": "1024x1024",
            "1:1": "1024x1024",
            "2:3": "832x1248",
            "3:2": "1248x832",
            "4:3": "1024x768",
            "3:4": "768x1024",
            "4:5": "896x1120",
            "5:4": "1120x896",
            "16:9": "1280x720",
            "9:16": "720x1280",
            "21:9": "1456x624",
        },
        "2K": {
            "": "2048x2048",
            "1:1": "2048x2048",
            "2:3": "1344x2016",
            "3:2": "2016x1344",
            "4:3": "2048x1536",
            "3:4": "1536x2048",
            "4:5": "1600x2000",
            "5:4": "2000x1600",
            "16:9": "2048x1152",
            "9:16": "1152x2048",
            "21:9": "2016x864",
        },
        "4K": {
            "": "2880x2880",
            "1:1": "2880x2880",
            "2:3": "2304x3456",
            "3:2": "3456x2304",
            "4:3": "2880x2160",
            "3:4": "2160x2880",
            "4:5": "2304x2880",
            "5:4": "2880x2304",
            "16:9": "3840x2160",
            "9:16": "2160x3840",
            "21:9": "3840x1648",
        },
    }
    normalized_resolution = resolution if resolution in size_map else "2K"
    return size_map[normalized_resolution].get(aspect_ratio, size_map[normalized_resolution][""])
