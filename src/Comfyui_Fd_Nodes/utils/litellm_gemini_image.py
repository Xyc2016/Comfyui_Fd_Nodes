import base64
import io
import json
import logging

import numpy as np
import requests
import torch
from PIL import Image

from ..config import FD_LITELLM_API_KEY, FD_LITELLM_BASE_URL
from .error_utils import ERROR_NSFW, ERROR_TIMEOUT, normalize_error_message
from .gemini_service import summarize_text

logger = logging.getLogger(__name__)


LITELLM_GEMINI_MODELS = {
    "gemini-3-pro-image-preview-siphonlab",
}


def should_use_litellm_gemini(model: str) -> bool:
    return str(model or "").strip() in LITELLM_GEMINI_MODELS


def tensor_to_base64(image_tensor) -> str:
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_tensor(data_url: str):
    if ";base64," in data_url:
        _, b64 = data_url.split(";base64,", 1)
    else:
        b64 = data_url
    img_bytes = base64.b64decode(b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def summarize_messages_for_log(messages):
    summarized_messages = []
    for message in messages:
        summarized_message = {"role": message.get("role", "")}
        content = message.get("content", [])
        if isinstance(content, list):
            text_parts = []
            image_count = 0
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text = part.get("text", "").strip()
                    if text:
                        text_parts.append(summarize_text(text))
                elif part.get("type") in {"image_url", "image"}:
                    image_count += 1
            if text_parts:
                summarized_message["text"] = "\n".join(text_parts)
            if image_count:
                summarized_message["image_count"] = image_count
        else:
            summarized_message["content"] = content
        summarized_messages.append(summarized_message)
    return summarized_messages


def summarize_response_for_log(result):
    summary = {"keys": list(result.keys())}
    choices = result.get("choices", [])
    summary["choice_count"] = len(choices)
    if not choices:
        return summary

    choice = choices[0]
    summary["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message", {})
    summary["message_keys"] = list(message.keys())

    images = message.get("images", [])
    if isinstance(images, list):
        summary["image_count"] = len(images)

    content = message.get("content", [])
    if isinstance(content, list):
        content_types = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type:
                    content_types.append(part_type)
        if content_types:
            summary["content_types"] = content_types
    elif isinstance(content, str):
        summary["content_length"] = len(content)

    return summary


def extract_image_from_response(result):
    choice = result["choices"][0]
    if choice.get("finish_reason") == "content_filter":
        raise RuntimeError(
            normalize_error_message(
                "内容被过滤 (content_filter)，请修改提示词或输入图片后重试",
                category=ERROR_NSFW,
            )
        )
    msg = choice["message"]

    if "images" in msg and msg["images"]:
        return msg["images"][0]["image_url"]["url"]

    content = msg.get("content", [])
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url":
                    return part["image_url"]["url"]
                if part.get("type") == "image":
                    return part.get("url") or part.get("data", "")

    if isinstance(content, str) and len(content) > 100:
        return content

    raise KeyError(f"无法从响应中提取图片，响应结构: {list(msg.keys())}")


def build_litellm_messages(prompt: str, image_b64_list: list[str], system_prompt: str = ""):
    messages = []
    if system_prompt.strip():
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })
    user_content = [{"type": "text", "text": prompt}]
    for b64 in image_b64_list:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    messages.append({"role": "user", "content": user_content})
    return messages


def call_litellm_gemini_image(
    *,
    messages,
    model: str,
    aspect_ratio,
    image_size,
    seed: int,
    out_request_id: str = "",
    base_url: str = "",
    api_key: str = "",
):
    final_base_url = (base_url or FD_LITELLM_BASE_URL or "").rstrip("/")
    final_api_key = api_key or FD_LITELLM_API_KEY
    if not final_base_url or final_base_url == "https://your-api-base-url":
        raise RuntimeError("未配置 base_url，请在环境变量 FD_LITELLM_BASE_URL 中设置")
    if not final_api_key or final_api_key == "your-api-key":
        raise RuntimeError("未配置 api_key，请在环境变量 FD_LITELLM_API_KEY 中设置")

    url = f"{final_base_url}/v1/chat/completions"
    payload = {
        "stream": False,
        "model": model,
        "messages": messages,
        "imageConfig": {"aspect_ratio": aspect_ratio, "image_size": image_size},
        "modalities": ["image"],
    }
    if out_request_id:
        payload["user"] = out_request_id
    logger.info(
        "Calling LiteLLM Gemini image API with payload=%s",
        {
            "url": url,
            "stream": payload["stream"],
            "model": payload["model"],
            "imageConfig": payload["imageConfig"],
            "modalities": payload["modalities"],
            "user": payload.get("user"),
            "seed": seed,
            "messages": summarize_messages_for_log(messages),
        },
    )

    response = None
    try:
        response = requests.post(
            url=url,
            headers={
                "Authorization": f"Bearer {final_api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=600,
        )
        if not response.ok:
            raise RuntimeError(f"API 请求失败: {response.status_code}\n{response.text[:1000]}")
        result = response.json()
        logger.info(
            "LiteLLM Gemini image API response summary: %s",
            {
                "status_code": response.status_code,
                **summarize_response_for_log(result),
            },
        )
        out_data_url = extract_image_from_response(result)
    except (KeyError, IndexError) as exc:
        response_text = response.text[:500] if response is not None else ""
        raise RuntimeError(normalize_error_message(f"解析响应失败: {exc}\n原始响应: {response_text}")) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="request timed out")
        ) from exc
    except Exception as exc:
        raise RuntimeError(normalize_error_message(exc)) from exc
    return base64_to_tensor(out_data_url)
