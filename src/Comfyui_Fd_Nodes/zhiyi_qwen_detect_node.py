import ast
import json
import logging
import re

import numpy as np
import requests
import torch
from PIL import Image

from .config import FD_LITELLM_API_KEY, FD_LITELLM_BASE_URL
from .utils.error_utils import ERROR_TIMEOUT, normalize_error_message
from .utils.litellm_gemini_image import tensor_to_base64
from .utils.logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BBox parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_text(raw: str) -> str:
    """Strip markdown fences and extract the JSON payload from model output."""
    text = raw.strip()

    # Handle ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # If model wrapped output in {"content": "..."} (string)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, str):
                text = inner
            elif isinstance(inner, list):
                return json.dumps(inner)
    except (json.JSONDecodeError, TypeError):
        pass

    # If model added explanation before/after the JSON array/object,
    # try to find the first [ or { and match to the end.
    if not text.startswith(("[", "{")):
        bracket_match = re.search(r"[\[{]", text)
        if bracket_match:
            text = text[bracket_match.start():]

    # If text after the JSON array/object has trailing content, strip it.
    # Track string/escape state to avoid premature truncation from ] or } inside labels.
    if text.startswith(("[", "{")):
        opener = text[0]
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        escape_next = False
        end_pos = -1
        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break
        if end_pos > 0:
            text = text[:end_pos]

    return text


def _normalize_box(box: list, img_width: int, img_height: int, coord_format: str = "auto") -> list[int]:
    """Convert a bbox to absolute pixel coordinates [x1,y1,x2,y2].

    coord_format controls how values are interpreted:
      "auto"   — detect from value range ([0,1], [0,1000], or pixel)
      "01"     — values in [0,1] normalized
      "1000"   — Qwen VL integers 0-1000
      "pixel"  — already absolute pixel coordinates
    """
    if len(box) < 4:
        raise ValueError(f"bbox must have >= 4 elements, got {len(box)}")

    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    if coord_format == "auto":
        coords = [x1, y1, x2, y2]
        max_val = max(abs(v) for v in coords)
        if max_val <= 1.0:
            coord_format = "01"
        elif max_val > 1000:
            coord_format = "pixel"
        else:
            all_near_int = all(abs(v - round(v)) < 0.01 for v in coords)
            coord_format = "1000" if all_near_int else "pixel"

    if coord_format == "01":
        x1, y1, x2, y2 = x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height
    elif coord_format == "1000":
        x1, y1, x2, y2 = (
            x1 / 1000 * img_width, y1 / 1000 * img_height,
            x2 / 1000 * img_width, y2 / 1000 * img_height,
        )
    # else "pixel": use as-is

    # Fix reversed boxes
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Clamp to image bounds
    x1 = max(0, min(round(x1), img_width))
    y1 = max(0, min(round(y1), img_height))
    x2 = max(0, min(round(x2), img_width))
    y2 = max(0, min(round(y2), img_height))

    return [x1, y1, x2, y2]


def _safe_score(value) -> float:
    """Parse a score value defensively, defaulting to 1.0."""
    try:
        s = float(value)
        if s != s:  # NaN check
            return 1.0
        return s
    except (TypeError, ValueError):
        return 1.0


def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    score_threshold: float = 0.0,
    coord_format: str = "auto",
) -> list[dict]:
    """Parse bounding boxes from model output text."""
    text = _extract_json_text(text)

    # Try json.loads first, then ast.literal_eval as fallback
    data = None
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(text)
            break
        except Exception:
            continue

    # Last resort: try to fix truncated JSON
    if data is None:
        try:
            end_idx = text.rfind("}") + 1
            if end_idx > 0:
                fixed = text[:end_idx] + "]"
                data = ast.literal_eval(fixed)
        except Exception:
            pass

    if data is None:
        logger.warning("Failed to parse bbox output: %s", text[:200])
        return []

    # Handle {"content": [...]} wrapper
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, list):
            data = inner
        elif isinstance(inner, str):
            try:
                data = json.loads(inner)
            except (json.JSONDecodeError, TypeError):
                try:
                    data = ast.literal_eval(inner)
                except Exception:
                    data = []
        else:
            data = []

    if not isinstance(data, list):
        logger.warning("Unexpected bbox data type: %s", type(data).__name__)
        return []

    items = []
    for item in data:
        if not isinstance(item, dict):
            # Item itself might be [x1,y1,x2,y2]
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                try:
                    box = _normalize_box(list(item), img_width, img_height, coord_format)
                    items.append({"bbox": box, "score": 1.0, "label": ""})
                except (ValueError, IndexError):
                    continue
            continue

        # Try common field names for bbox
        raw_box = None
        for field in ("bbox_2d", "bbox", "box_2d", "box"):
            if field in item:
                raw_box = item[field]
                break

        if raw_box is None:
            continue

        label = str(item.get("label", ""))
        score = _safe_score(item.get("score", 1.0))

        try:
            box = _normalize_box(list(raw_box), img_width, img_height, coord_format)
        except (ValueError, IndexError) as exc:
            logger.warning("Skipping invalid bbox %s: %s", raw_box, exc)
            continue

        if score >= score_threshold:
            items.append({"bbox": box, "score": score, "label": label})

    items.sort(key=lambda x: x["score"], reverse=True)
    return items


# ---------------------------------------------------------------------------
# URL helper
# ---------------------------------------------------------------------------

def _build_chat_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_detection_prompt(target: str) -> str:
    return (
        f'Detect all visible instances of "{target}" in this image.\n'
        "Return ONLY a JSON array. No markdown, no explanation.\n"
        'Each item: {"bbox_2d":[x1,y1,x2,y2],"label":"...","score":0.0}\n'
        "Coordinates: normalized integers 0-1000 relative to image size.\n"
        "Order: [x1,y1,x2,y2]. If none found, return []."
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class ZhiYiQwenDetectNode:
    """Use Qwen3-VL-Flash API to detect objects and output bounding boxes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target": ("STRING", {"default": "object"}),
                "model": ("STRING", {"default": "qwen3-vl-flash"}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_selection": ("STRING", {"default": "all"}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "coordinate_format": (["auto", "1000", "01", "pixel"],),
            },
        }

    RETURN_TYPES = ("JSON", "BBOX", "BBOXES")
    RETURN_NAMES = ("json", "bboxes", "bboxes_for_sam2")
    FUNCTION = "detect"
    CATEGORY = "知衣/目标检测"
    OUTPUT_NODE = False

    def detect(
        self,
        image,
        target: str,
        model: str = "qwen3-vl-flash",
        score_threshold: float = 0.0,
        bbox_selection: str = "all",
        merge_boxes: bool = False,
        node_switch: int = 0,
        coordinate_format: str = "auto",
    ):
        if node_switch == 1:
            return ("[]", [], [[]])

        # ComfyUI IMAGE is always a torch tensor
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(image)}")

        if image.ndim == 4:
            img_tensor = image[0]
        else:
            img_tensor = image
        arr = (img_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr).convert("RGB")
        img_width, img_height = pil_img.size

        # Encode image
        b64 = tensor_to_base64(image)
        data_url = f"data:image/png;base64,{b64}"

        # Build messages — image before text (per OpenAI vision API convention)
        prompt = _build_detection_prompt(target)
        messages = [
            {"role": "system", "content": "You are a helpful visual grounding assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Call API
        base_url = FD_LITELLM_BASE_URL or ""
        api_key = FD_LITELLM_API_KEY or ""
        if not base_url:
            raise RuntimeError("未配置 FD_LITELLM_BASE_URL")
        if not api_key:
            raise RuntimeError("未配置 FD_LITELLM_API_KEY")

        url = _build_chat_url(base_url)
        payload = {
            "stream": False,
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
            "enable_thinking": False,
        }

        logger.info(
            "Qwen detect API call: url=%s model=%s target=%s img_size=%s",
            url, model, target, (img_width, img_height),
        )

        response = None
        try:
            response = requests.post(
                url=url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=120,
            )
            if not response.ok:
                raise RuntimeError(
                    f"API 请求失败: {response.status_code}\n{response.text[:1000]}"
                )
            result = response.json()

            choice = result["choices"][0]
            finish_reason = choice.get("finish_reason", "")
            output_text = choice["message"]["content"]
            if isinstance(output_text, list):
                output_text = "".join(
                    p.get("text", "") for p in output_text if isinstance(p, dict)
                )

            if finish_reason == "length":
                logger.warning("Model hit max_tokens limit, output may be truncated")

            logger.info("Qwen detect response: finish_reason=%s output_len=%d", finish_reason, len(output_text))

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="Qwen detect request timed out")
            ) from exc
        except (KeyError, IndexError) as exc:
            resp_text = response.text[:500] if response is not None else ""
            raise RuntimeError(
                normalize_error_message(f"解析响应失败: {exc}\n原始响应: {resp_text}")
            ) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(normalize_error_message(exc)) from exc

        # Parse boxes
        items = parse_boxes(output_text, img_width, img_height, score_threshold, coordinate_format)

        # Apply bbox selection
        selection = bbox_selection.strip().lower()
        boxes = items
        if selection not in ("all", "") and selection:
            idxs = []
            for part in selection.replace(",", " ").split():
                try:
                    idxs.append(int(part))
                except ValueError:
                    continue
            boxes = [boxes[i] for i in idxs if 0 <= i < len(boxes)]

        # Merge boxes
        if merge_boxes and boxes:
            x1 = min(b["bbox"][0] for b in boxes)
            y1 = min(b["bbox"][1] for b in boxes)
            x2 = max(b["bbox"][2] for b in boxes)
            y2 = max(b["bbox"][3] for b in boxes)
            score = max(b["score"] for b in boxes)
            label = boxes[0].get("label", target)
            boxes = [{"bbox": [x1, y1, x2, y2], "score": score, "label": label}]

        # Build outputs
        json_output = json.dumps(
            [{"bbox_2d": b["bbox"], "label": b.get("label", target)} for b in boxes],
            ensure_ascii=False,
        )
        bboxes_only = [b["bbox"] for b in boxes]
        # BBOXES for SAM2: always wrap in batch list, even when empty
        bboxes_for_sam2 = [bboxes_only]

        return (json_output, bboxes_only, bboxes_for_sam2)


class ZhiYiBBoxesToSAM2:
    """Convert BBOX list to BBOXES format for SAM2 nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bboxes": ("BBOX",)}}

    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("sam2_bboxes",)
    FUNCTION = "convert"
    CATEGORY = "知衣/目标检测"

    def convert(self, bboxes):
        if not isinstance(bboxes, list):
            raise ValueError("bboxes must be a list")
        # Already batched
        if (
            bboxes
            and isinstance(bboxes[0], (list, tuple))
            and bboxes[0]
            and isinstance(bboxes[0][0], (list, tuple))
        ):
            return (bboxes,)
        return ([bboxes],)
