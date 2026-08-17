import json
import logging
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageFilter

from .utils.logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)


CANDIDATE_COLORS = [
    {"name": "亮青蓝色", "rgb": (0, 170, 255)},
    {"name": "亮蓝色", "rgb": (0, 120, 255)},
    {"name": "亮绿色", "rgb": (0, 210, 80)},
    {"name": "亮洋红", "rgb": (255, 0, 170)},
    {"name": "亮紫红色", "rgb": (255, 40, 120)},
    {"name": "亮紫色", "rgb": (150, 70, 255)},
    {"name": "亮湖蓝色", "rgb": (0, 210, 210)},
    {"name": "亮橙色", "rgb": (255, 120, 0)},
    {"name": "亮黄色", "rgb": (255, 235, 0)},
    {"name": "亮白色", "rgb": (245, 245, 245)},
]


def _clamp(value, low=0, high=255):
    return max(low, min(high, int(round(value))))


def _image_tensor_to_pil(image_tensor):
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image_array = image_tensor.detach().cpu().numpy()
    image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    channels = image_array.shape[2] if image_array.ndim == 3 else 0
    if channels == 4:
        return Image.fromarray(image_array, mode="RGBA")
    return Image.fromarray(image_array, mode="RGB")


def _pil_to_image_tensor(image):
    image_array = np.asarray(image.convert("RGBA" if image.mode == "RGBA" else "RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


def _mask_to_tensor(mask_image):
    mask_array = np.asarray(mask_image.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(mask_array).unsqueeze(0)


def _quantized_palette(image, colors=10):
    quantized = image.quantize(colors=colors, method=Image.MEDIANCUT)
    palette = quantized.getpalette()
    counts = Counter(quantized.getdata())
    result = []
    for idx, count in counts.most_common(colors):
        rgb = tuple(palette[idx * 3: idx * 3 + 3])
        result.append((count, rgb))
    return result


def _color_distance(rgb1, rgb2):
    return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5


def choose_background_pair_from_image(image):
    image = image.convert("RGB")
    full = image.resize((256, 256))
    width, height = image.size
    center = image.crop(
        (
            int(width * 0.2),
            int(height * 0.18),
            int(width * 0.8),
            int(height * 0.72),
        )
    ).resize((256, 256))

    full_palette = _quantized_palette(full, colors=10)
    center_palette = _quantized_palette(center, colors=10)
    full_total = sum(count for count, _ in full_palette)
    center_total = sum(count for count, _ in center_palette)

    all_candidates = []
    for candidate in CANDIDATE_COLORS:
        rgb = candidate["rgb"]
        full_mean = sum(count * _color_distance(rgb, ref_rgb) for count, ref_rgb in full_palette) / full_total
        center_mean = sum(count * _color_distance(rgb, ref_rgb) for count, ref_rgb in center_palette) / center_total
        full_min = min(_color_distance(rgb, ref_rgb) for _, ref_rgb in full_palette)
        center_min = min(_color_distance(rgb, ref_rgb) for _, ref_rgb in center_palette)
        score = center_mean * 0.45 + full_mean * 0.15 + center_min * 0.3 + full_min * 0.1
        all_candidates.append(
            {
                "name": candidate["name"],
                "rgb": list(rgb),
                "center_mean": round(center_mean, 6),
                "full_mean": round(full_mean, 6),
                "center_min": round(center_min, 6),
                "full_min": round(full_min, 6),
                "score": round(score, 6),
            }
        )

    def select_with_threshold(threshold):
        accepted = []
        rejected = []
        for item in all_candidates:
            cloned = dict(item)
            if cloned["center_min"] < threshold:
                cloned["rejected_reason"] = "center_min_below_threshold_{}".format(threshold)
                rejected.append(cloned)
            else:
                accepted.append(cloned)
        accepted.sort(key=lambda x: x["score"], reverse=True)
        return accepted, rejected

    strategy = "strict"
    accepted, rejected = select_with_threshold(190.0)
    if len(accepted) < 2:
        strategy = "relaxed"
        accepted, rejected = select_with_threshold(160.0)
    if len(accepted) < 2:
        strategy = "fallback_top2"
        accepted = sorted((dict(item) for item in all_candidates), key=lambda x: x["score"], reverse=True)
        rejected = []

    bg1_candidates = [item for item in accepted if item["name"] != "亮白色"]
    if not bg1_candidates:
        bg1_candidates = accepted

    best_pair = None
    best_pair_score = None
    for bg1 in bg1_candidates:
        for bg2 in accepted:
            if bg1["name"] == bg2["name"]:
                continue
            if bg2["name"] == "亮白色":
                continue
            pair_gap = _color_distance(tuple(bg1["rgb"]), tuple(bg2["rgb"]))
            pair_score = bg1["score"] + bg2["score"] + pair_gap * 0.45
            if best_pair_score is None or pair_score > best_pair_score:
                best_pair = (bg1, bg2)
                best_pair_score = pair_score

    if best_pair is None or best_pair_score is None:
        raise RuntimeError("无法为图片选择双背景色")

    explain = {
        "ranked_candidates": accepted,
        "rejected_candidates": rejected,
        "pair_gap": round(_color_distance(tuple(best_pair[0]["rgb"]), tuple(best_pair[1]["rgb"])), 6),
        "pair_score": round(best_pair_score, 6),
        "selection_strategy": strategy,
        "bg1_white_forbidden": True,
        "bg2_white_forbidden": True,
        "strict_threshold": 190.0,
        "relaxed_threshold": 160.0,
    }
    return best_pair[0], best_pair[1], explain


def _sample_background_color(image, sample_span=24):
    rgb = image.convert("RGB")
    width, height = rgb.size
    span = max(1, min(sample_span, width, height))

    points = []
    for x in range(span):
        for y in range(span):
            points.append((x, y))
            points.append((width - 1 - x, y))
            points.append((x, height - 1 - y))
            points.append((width - 1 - x, height - 1 - y))

    total_r = total_g = total_b = 0
    for x, y in points:
        r, g, b = rgb.getpixel((x, y))
        total_r += r
        total_g += g
        total_b += b

    count = max(1, len(points))
    return (
        _clamp(total_r / count),
        _clamp(total_g / count),
        _clamp(total_b / count),
    )


def _estimate_translation_offset(img_a, img_b, max_offset=20, downsample_step=4):
    a = np.asarray(img_a.convert("L")).astype(np.float32)
    b = np.asarray(img_b.convert("L")).astype(np.float32)

    def gradient(arr):
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
        gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
        return np.abs(gx) + np.abs(gy)

    ga = gradient(a)[::downsample_step, ::downsample_step]
    gb = gradient(b)[::downsample_step, ::downsample_step]

    best_score = None
    best_dx = 0
    best_dy = 0
    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            y1s = max(0, dy)
            y1e = min(ga.shape[0], gb.shape[0] + dy)
            x1s = max(0, dx)
            x1e = min(ga.shape[1], gb.shape[1] + dx)
            y2s = max(0, -dy)
            y2e = min(gb.shape[0], ga.shape[0] - dy)
            x2s = max(0, -dx)
            x2e = min(gb.shape[1], ga.shape[1] - dx)
            if y1e - y1s < 20 or x1e - x1s < 20:
                continue
            score = float(np.mean(np.abs(ga[y1s:y1e, x1s:x1e] - gb[y2s:y2e, x2s:x2e])))
            if best_score is None or score < best_score:
                best_score = score
                best_dx = dx * downsample_step
                best_dy = dy * downsample_step
    return best_dx, best_dy


def _translate_image(image, dx, dy, fill_rgb):
    src = np.asarray(image.convert("RGB"))
    out = np.zeros_like(src)
    out[..., 0] = fill_rgb[0]
    out[..., 1] = fill_rgb[1]
    out[..., 2] = fill_rgb[2]

    height, width = src.shape[:2]
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    copied_width = min(width - src_x0, width - dst_x0)
    copied_height = min(height - src_y0, height - dst_y0)
    if copied_width > 0 and copied_height > 0:
        out[dst_y0: dst_y0 + copied_height, dst_x0: dst_x0 + copied_width] = src[
            src_y0: src_y0 + copied_height,
            src_x0: src_x0 + copied_width,
        ]
    return Image.fromarray(out, mode="RGB")


def _crop_to_common_size(img_a, img_b):
    if img_a.size == img_b.size:
        return img_a, img_b

    target_width = min(img_a.size[0], img_b.size[0])
    target_height = min(img_a.size[1], img_b.size[1])
    crop_box = (0, 0, target_width, target_height)
    return img_a.crop(crop_box), img_b.crop(crop_box)


def _estimate_alpha_and_foreground(img_a, img_b, bg_a, bg_b, alpha_floor=0.02):
    a_arr = np.asarray(img_a.convert("RGB")).astype(np.float32)
    b_arr = np.asarray(img_b.convert("RGB")).astype(np.float32)

    bg_delta = np.asarray(bg_a, dtype=np.float32) - np.asarray(bg_b, dtype=np.float32)
    bg_norm_sq = max(1e-6, float(np.dot(bg_delta, bg_delta)))

    diff = a_arr - b_arr
    alpha = 1.0 - (np.sum(diff * bg_delta[None, None, :], axis=2) / bg_norm_sq)
    alpha = np.clip(alpha, 0.0, 1.0)

    keep_mask = alpha > alpha_floor
    safe_alpha = np.maximum(alpha, 1e-6)
    fg = (
        a_arr - (1.0 - alpha[..., None]) * np.asarray(bg_a, dtype=np.float32)[None, None, :]
    ) / safe_alpha[..., None]
    fg = np.clip(fg, 0.0, 255.0)
    fg[~keep_mask] = 0.0

    alpha_uint8 = np.where(keep_mask, np.clip(np.round(alpha * 255.0), 0.0, 255.0), 0.0).astype(np.uint8)
    foreground = np.dstack([fg.astype(np.uint8), alpha_uint8])
    return Image.fromarray(alpha_uint8, mode="L"), Image.fromarray(foreground, mode="RGBA")


def _estimate_dual_foregrounds(img_a, img_b, alpha_mask, bg_a, bg_b):
    a_arr = np.asarray(img_a.convert("RGB")).astype(np.float32)
    b_arr = np.asarray(img_b.convert("RGB")).astype(np.float32)
    alpha_arr = np.asarray(alpha_mask.convert("L")).astype(np.float32)

    alpha_ratio = alpha_arr / 255.0
    keep_mask = alpha_arr > 0.0
    safe_alpha = np.maximum(alpha_ratio, 1e-6)

    fg_a = (a_arr - (1.0 - alpha_ratio[..., None]) * np.asarray(bg_a, dtype=np.float32)[None, None, :]) / safe_alpha[..., None]
    fg_b = (b_arr - (1.0 - alpha_ratio[..., None]) * np.asarray(bg_b, dtype=np.float32)[None, None, :]) / safe_alpha[..., None]
    fg_a = np.clip(fg_a, 0.0, 255.0)
    fg_b = np.clip(fg_b, 0.0, 255.0)
    fg_a[~keep_mask] = 0.0
    fg_b[~keep_mask] = 0.0

    out_a = np.dstack([fg_a.astype(np.uint8), alpha_arr.astype(np.uint8)])
    out_b = np.dstack([fg_b.astype(np.uint8), alpha_arr.astype(np.uint8)])
    return Image.fromarray(out_a, mode="RGBA"), Image.fromarray(out_b, mode="RGBA")


def _suppress_background_spill(rgba, bg_a, bg_b, alpha_floor=0.02):
    arr = np.asarray(rgba.convert("RGBA")).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    alpha_ratio = alpha / 255.0

    bg_a_arr = np.asarray(bg_a, dtype=np.float32)
    bg_b_arr = np.asarray(bg_b, dtype=np.float32)
    dist_a = np.linalg.norm(rgb - bg_a_arr[None, None, :], axis=2)
    dist_b = np.linalg.norm(rgb - bg_b_arr[None, None, :], axis=2)
    bg_dist = np.minimum(dist_a, dist_b)

    new_alpha = alpha.copy()
    new_alpha = np.where(bg_dist < 28.0, np.round(alpha * 0.15), new_alpha)
    new_alpha = np.where((bg_dist >= 28.0) & (bg_dist < 42.0), np.round(alpha * 0.45), new_alpha)
    new_alpha = np.where((bg_dist >= 42.0) & (bg_dist < 60.0), np.round(alpha * 0.75), new_alpha)

    keep_mask = (alpha > 0.0) & (alpha_ratio > alpha_floor) & (new_alpha > 0.0)
    out = np.zeros_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    out[..., 3] = np.where(keep_mask, np.clip(new_alpha, 0.0, 255.0), 0.0).astype(np.uint8)
    out[~keep_mask, :3] = 0
    return Image.fromarray(out, mode="RGBA")


def _build_clean_alpha(alpha_mask, low_cut=56, blur_radius=0.8):
    alpha = alpha_mask.convert("L")
    thresholded = alpha.point(lambda px: 0 if px < low_cut else px, mode="L")
    if blur_radius > 0:
        thresholded = thresholded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return thresholded.point(lambda px: 0 if px < 72 else 255 if px > 200 else px, mode="L")


def _build_agreement_rgba(fg_from_a, fg_from_b, alpha_mask, agreement_soft=20.0, agreement_hard=60.0):
    arr_a = np.asarray(fg_from_a.convert("RGBA")).astype(np.float32)
    arr_b = np.asarray(fg_from_b.convert("RGBA")).astype(np.float32)
    alpha = np.asarray(alpha_mask.convert("L")).astype(np.float32)

    rgb_a = arr_a[..., :3]
    rgb_b = arr_b[..., :3]
    rgb_mean = (rgb_a + rgb_b) / 2.0
    diff = np.linalg.norm(rgb_a - rgb_b, axis=2)

    out_alpha = alpha.copy()
    alpha_ratio = alpha / 255.0
    active = alpha > 0.0
    soft_mask = active & (diff > agreement_soft) & (diff < agreement_hard)
    hard_mask = active & (diff >= agreement_hard)

    if np.any(hard_mask):
        hard_scale = np.where(alpha_ratio[hard_mask] < 0.30, 0.0, 0.25)
        out_alpha[hard_mask] = alpha[hard_mask] * hard_scale

    if np.any(soft_mask):
        ratio = (diff[soft_mask] - agreement_soft) / max(1e-6, agreement_hard - agreement_soft)
        scale = 1.0 - 0.85 * ratio
        low_mask = alpha_ratio[soft_mask] < 0.25
        mid_mask = (alpha_ratio[soft_mask] >= 0.25) & (alpha_ratio[soft_mask] < 0.45)
        scale[low_mask] *= 0.45
        scale[mid_mask] *= 0.70
        out_alpha[soft_mask] = alpha[soft_mask] * scale

    out = np.dstack([np.clip(rgb_mean, 0, 255), np.clip(out_alpha, 0, 255)]).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _apply_alpha_to_rgba(rgba, alpha_mask):
    result = rgba.convert("RGBA").copy()
    result.putalpha(alpha_mask.convert("L"))
    return result


def _mask_tensor_to_pil(alpha_mask, size=None):
    """Convert a ComfyUI MASK tensor to a PIL 'L' image.

    ComfyUI masks are usually float tensors in [0, 1] with shape (B, H, W)
    or (H, W).  Accept both and tolerate masks that are already in uint8
    [0, 255] range.
    """
    if alpha_mask.ndim == 4 and alpha_mask.shape[1] == 1:
        alpha_mask = alpha_mask[:, 0, :, :]
    if alpha_mask.ndim == 4 and alpha_mask.shape[-1] == 1:
        alpha_mask = alpha_mask[..., 0]
    if alpha_mask.ndim == 3:
        alpha_mask = alpha_mask[0]
    mask_np = alpha_mask.detach().cpu().numpy().astype(np.float32)
    if mask_np.max() > 1.0:
        mask_np = mask_np / 255.0
    mask_np = np.clip(mask_np, 0.0, 1.0)
    mask_uint8 = np.round(mask_np * 255.0).astype(np.uint8)
    mask_image = Image.fromarray(mask_uint8, mode="L")
    if size is not None and mask_image.size != size:
        mask_image = mask_image.resize(size, Image.BILINEAR)
    return mask_image


def _build_unified_rgba(rgba_soft, rgba_agreement, rgba_clean):
    soft = np.asarray(rgba_soft.convert("RGBA")).astype(np.float32)
    agreement = np.asarray(rgba_agreement.convert("RGBA")).astype(np.float32)
    clean = np.asarray(rgba_clean.convert("RGBA")).astype(np.float32)

    soft_rgb = soft[..., :3]
    agreement_rgb = agreement[..., :3]
    clean_rgb = clean[..., :3]
    soft_alpha = soft[..., 3]
    agreement_alpha = agreement[..., 3]
    clean_alpha = clean[..., 3]

    height, width = soft_alpha.shape
    yy, xx = np.mgrid[0:height, 0:width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.sqrt(((yy - center_y) / max(height, 1)) ** 2 + ((xx - center_x) / max(width, 1)) ** 2)
    center_weight = np.clip(1.0 - radius * 2.2, 0.0, 1.0)

    sure_fg = agreement_alpha >= 220.0
    sure_bg = clean_alpha <= 8.0
    ambiguous = (~sure_fg) & (~sure_bg)
    inner_ambiguous = ambiguous & (center_weight >= 0.16)
    outer_ambiguous = ambiguous & (~inner_ambiguous)

    out_alpha = clean_alpha.copy()
    out_rgb = clean_rgb.copy()

    out_alpha[sure_fg] = np.maximum(out_alpha[sure_fg], 245.0)
    out_rgb[sure_fg] = agreement_rgb[sure_fg]
    out_alpha[sure_bg] = 0.0

    if np.any(inner_ambiguous):
        inner_alpha = np.maximum(clean_alpha[inner_ambiguous], agreement_alpha[inner_ambiguous] * 0.85)
        inner_alpha = np.maximum(inner_alpha, soft_alpha[inner_ambiguous] * 0.92)
        out_alpha[inner_ambiguous] = np.clip(inner_alpha, 0.0, 255.0)
        out_rgb[inner_ambiguous] = 0.65 * agreement_rgb[inner_ambiguous] + 0.35 * clean_rgb[inner_ambiguous]

    if np.any(outer_ambiguous):
        outer_alpha = 0.55 * soft_alpha[outer_ambiguous] + 0.45 * clean_alpha[outer_ambiguous]
        out_alpha[outer_ambiguous] = np.clip(outer_alpha, 0.0, 255.0)
        out_rgb[outer_ambiguous] = 0.7 * soft_rgb[outer_ambiguous] + 0.3 * clean_rgb[outer_ambiguous]

    return Image.fromarray(
        np.dstack([np.clip(out_rgb, 0, 255), np.clip(out_alpha, 0, 255)]).astype(np.uint8),
        mode="RGBA",
    )


def generate_rgba_from_dual_background_images(img_a, img_b, sample_span=24, alpha_floor=0.02):
    img_a, img_b = _crop_to_common_size(img_a, img_b)

    bg_a = _sample_background_color(img_a, sample_span=sample_span)
    bg_b = _sample_background_color(img_b, sample_span=sample_span)

    dx, dy = _estimate_translation_offset(img_a, img_b)
    if dx or dy:
        img_b = _translate_image(img_b, dx, dy, bg_b)
        bg_b = _sample_background_color(img_b, sample_span=sample_span)

    alpha_mask, rgba_raw = _estimate_alpha_and_foreground(img_a, img_b, bg_a, bg_b, alpha_floor=alpha_floor)
    fg_from_a, fg_from_b = _estimate_dual_foregrounds(img_a, img_b, alpha_mask, bg_a, bg_b)
    rgba_soft = _suppress_background_spill(rgba_raw, bg_a, bg_b, alpha_floor=alpha_floor)
    rgba_agreement = _build_agreement_rgba(fg_from_a, fg_from_b, rgba_soft.getchannel("A"))
    rgba_agreement = _suppress_background_spill(rgba_agreement, bg_a, bg_b, alpha_floor=alpha_floor)
    rgba_clean = _apply_alpha_to_rgba(rgba_soft, _build_clean_alpha(rgba_soft.getchannel("A")))
    final_rgba = _build_unified_rgba(rgba_soft, rgba_agreement, rgba_clean)
    final_alpha = final_rgba.getchannel("A")

    meta = {
        "bg_a_rgb": list(bg_a),
        "bg_b_rgb": list(bg_b),
        "estimated_dx": int(dx),
        "estimated_dy": int(dy),
        "sample_span": int(sample_span),
        "alpha_floor": float(alpha_floor),
    }
    return final_rgba, final_alpha, meta, rgba_soft, rgba_agreement


class PatternChooseBackgroundPair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("bg1_name", "bg2_name", "bg1_rgb_json", "bg2_rgb_json", "selection_json")
    FUNCTION = "execute"
    CATEGORY = "essentials/pattern extraction"

    def execute(self, image):
        pil_image = _image_tensor_to_pil(image)
        bg1, bg2, explain = choose_background_pair_from_image(pil_image)
        logger.info(
            "PatternChooseBackgroundPair selected bg1=%s rgb=%s, bg2=%s rgb=%s, strategy=%s",
            bg1["name"], bg1["rgb"], bg2["name"], bg2["rgb"], explain.get("selection_strategy"),
        )
        return (
            bg1["name"],
            bg2["name"],
            json.dumps(bg1["rgb"], ensure_ascii=False),
            json.dumps(bg2["rgb"], ensure_ascii=False),
            json.dumps(explain, ensure_ascii=False),
        )


class PatternDualBackgroundToRGBA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "sample_span": ("INT", {"default": 24, "min": 1, "max": 256, "step": 1}),
                "alpha_floor": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (
        "rgba_image",
        "alpha_mask",
        "meta_json",
        "rgba_soft",
        "rgba_agreement",
        "preferred_rgba",
        "preferred_alpha_mask",
    )
    FUNCTION = "execute"
    CATEGORY = "essentials/pattern extraction"

    def execute(self, image_a, image_b, sample_span, alpha_floor):
        batch_size = max(image_a.shape[0], image_b.shape[0])
        rgba_list = []
        alpha_list = []
        meta_list = []
        rgba_soft_list = []
        rgba_agreement_list = []

        for i in range(batch_size):
            frame_a = image_a[min(i, image_a.shape[0] - 1): min(i, image_a.shape[0] - 1) + 1]
            frame_b = image_b[min(i, image_b.shape[0] - 1): min(i, image_b.shape[0] - 1) + 1]
            pil_a = _image_tensor_to_pil(frame_a)
            pil_b = _image_tensor_to_pil(frame_b)
            rgba_image, alpha_mask, meta, rgba_soft, rgba_agreement = generate_rgba_from_dual_background_images(
                pil_a,
                pil_b,
                sample_span=sample_span,
                alpha_floor=alpha_floor,
            )
            logger.info(
                "PatternDualBackgroundToRGBA frame %s/%s bg_a=%s bg_b=%s dx=%s dy=%s",
                i + 1, batch_size, meta["bg_a_rgb"], meta["bg_b_rgb"],
                meta["estimated_dx"], meta["estimated_dy"],
            )
            rgba_list.append(_pil_to_image_tensor(rgba_image))
            alpha_list.append(_mask_to_tensor(alpha_mask))
            meta_list.append(meta)
            rgba_soft_list.append(_pil_to_image_tensor(rgba_soft))
            rgba_agreement_list.append(_pil_to_image_tensor(rgba_agreement))

        rgba_batch = torch.cat(rgba_list, dim=0)
        alpha_batch = torch.cat(alpha_list, dim=0)
        return (
            rgba_batch,
            alpha_batch,
            json.dumps(meta_list, ensure_ascii=False),
            torch.cat(rgba_soft_list, dim=0),
            torch.cat(rgba_agreement_list, dim=0),
            rgba_batch,
            alpha_batch,
        )


class PatternApplyAlphaToImage:
    """Apply a ComfyUI MASK as the alpha channel of an image.

    The node converts the input image to RGBA, replaces/sets the alpha
    channel from the supplied mask, optionally zeros the RGB of fully
    transparent pixels, and returns the composited RGBA image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_mask": ("MASK",),
                "zero_transparent_rgb": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba_image",)
    FUNCTION = "execute"
    CATEGORY = "essentials/pattern extraction"

    def execute(self, image, alpha_mask, zero_transparent_rgb=True):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        batch_size = image.shape[0]
        output_frames = []

        for i in range(batch_size):
            frame = image[i : i + 1]
            pil_image = _image_tensor_to_pil(frame).convert("RGBA")
            frame_mask = alpha_mask
            if alpha_mask.ndim == 3 and alpha_mask.shape[0] == batch_size:
                frame_mask = alpha_mask[i : i + 1]
            elif alpha_mask.ndim == 3 and alpha_mask.shape[0] > 1 and batch_size == 1:
                frame_mask = alpha_mask[0:1]
            mask_image = _mask_tensor_to_pil(frame_mask, size=pil_image.size)

            result = pil_image.copy()
            result.putalpha(mask_image)

            if zero_transparent_rgb:
                arr = np.array(result)
                alpha_channel = arr[..., 3]
                arr[alpha_channel == 0, :3] = 0
                result = Image.fromarray(arr, mode="RGBA")

            output_frames.append(_pil_to_image_tensor(result))

        return (torch.cat(output_frames, dim=0),)


PATTERN_CLASS_MAPPINGS = {
    "PatternChooseBackgroundPair+": PatternChooseBackgroundPair,
    "PatternDualBackgroundToRGBA+": PatternDualBackgroundToRGBA,
    "PatternApplyAlphaToImage+": PatternApplyAlphaToImage,
}


PATTERN_NAME_MAPPINGS = {
    "PatternChooseBackgroundPair+": "🔧 Pattern Choose Background Pair",
    "PatternDualBackgroundToRGBA+": "🔧 Pattern Dual Background To RGBA",
    "PatternApplyAlphaToImage+": "🔧 Pattern Apply Alpha To Image",
}
