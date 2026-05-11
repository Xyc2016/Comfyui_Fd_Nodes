import requests
import json
import base64
import random
import numpy as np
import os
from PIL import Image
import io
import torch
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config_manager import load_config
from .utils.logging_utils import configure_default_logging
import logging
from .config import (
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
)

configure_default_logging()
logger = logging.getLogger(__name__)


class ZhiYiImageToImageNode:
    """知衣图生图节点 - 最多6张输入图片，batch_size 控制重复请求次数，支持提示词列表并发"""

    MODELS = [
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image-preview",
        "gemini-3-pro-image-preview-official",
    ]

    ASPECT_RATIOS = ["", "1:1", "16:9", "9:16", "4:3", "3:4"]
    IMAGE_SIZES = ["4K", "2K", "1080P", "720P"]
    SEED_MODES = ["随机种子", "固定种子"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "将裤子变为老欧美风格的牛仔裤",
                    "multiline": True,
                }),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": ""}),
                "image_size": (cls.IMAGE_SIZES, {"default": "2K"}),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "seed_mode": (cls.SEED_MODES, {"default": "随机种子"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                }),
            },
            "optional": {
                "node_switch": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "number",
                }),
                "out_request_id": ("STRING", {
                    "default": "default",
                    "tooltip": "FD out_request_id for generation",
                }),
                "prompt_list": ("LIST",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
    FUNCTION = "generate"
    CATEGORY = "知衣/图生图"
    OUTPUT_NODE = False

    def _tensor_to_base64(self, image_tensor):
        arr = (image_tensor[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _base64_to_tensor(self, data_url):
        if ";base64," in data_url:
            _, b64 = data_url.split(";base64,", 1)
        else:
            b64 = data_url
        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _summarize_messages_for_log(self, messages):
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
                            text_parts.append(text)
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

    def _summarize_response_for_log(self, result):
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

    def _extract_image_from_response(self, result):
        choice = result["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            raise RuntimeError("内容被过滤 (content_filter)，请修改提示词或输入图片后重试")
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

    MODELS_NO_SEED = {"gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"}

    def _single_request(self, url, api_key, messages, model, aspect_ratio, image_size, seed, out_request_id="default"):
        payload = {
            "stream": False,
            "model": model,
            "messages": messages,
            "imageConfig": {"aspect_ratio": aspect_ratio, "image_size": image_size},
            "modalities": ["image"],
            "user": out_request_id,
        }
        if model not in self.MODELS_NO_SEED:
            payload["seed"] = seed
        logger.info(
            "Calling ZhiYi image-to-image API with payload=%s",
            {
                "url": url,
                "stream": payload["stream"],
                "model": payload["model"],
                "imageConfig": payload["imageConfig"],
                "modalities": payload["modalities"],
                "user": payload["user"],
                "seed": payload.get("seed"),
                "messages": self._summarize_messages_for_log(messages),
            },
        )
        response = requests.post(
            url=url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=600,
        )
        if not response.ok:
            raise RuntimeError(f"API 请求失败: {response.status_code}\n{response.text[:1000]}")
        result = response.json()
        logger.info(
            "ZhiYi image-to-image API response summary: %s",
            {
                "status_code": response.status_code,
                **self._summarize_response_for_log(result),
            },
        )
        try:
            out_data_url = self._extract_image_from_response(result)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"解析响应失败: {e}\n原始响应: {response.text[:500]}")
        return self._base64_to_tensor(out_data_url)

    def _build_messages(self, prompt, image_b64_list, system_prompt):
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

    def _run_concurrent(self, tasks, label="任务"):
        """tasks: list of (idx, callable, args)，并发执行，返回 {idx: result}"""
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                executor.submit(fn, *args): idx
                for idx, fn, args in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(
                        "[知衣图生图] %s 第 %s 个失败，已跳过: %s: %s",
                        label,
                        idx + 1,
                        type(e).__name__,
                        e,
                    )
                    traceback.print_exc()
        return results

    def generate(self, image_1, prompt, model, aspect_ratio, image_size,
                 batch_size=1, seed_mode="随机种子", seed=0,
                 node_switch=0, out_request_id="default", prompt_list=None,
                 image_2=None, image_3=None, image_4=None,
                 image_5=None, image_6=None, system_prompt=""):
        if node_switch == 1:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, 0)

        cfg = load_config()
        base_url = (FD_LITELLM_BASE_URL or cfg["base_url"]).rstrip("/")
        api_key = FD_LITELLM_API_KEY or cfg["api_key"]
        url = f"{base_url}/v1/chat/completions"

        actual_seed = random.randint(0, 2147483647) if seed_mode == "随机种子" else seed

        # 预先编码所有图片（共享，避免重复编码）
        image_tensors = [t for t in [image_1, image_2, image_3, image_4, image_5, image_6] if t is not None]
        image_b64_list = [self._tensor_to_base64(t) for t in image_tensors]

        # 有效提示词列表；字符串输入按单条 prompt 处理，避免被逐字符拆分
        if isinstance(prompt_list, str):
            prompt_candidates = [prompt_list]
        else:
            prompt_candidates = prompt_list or []

        prompts = [p for p in prompt_candidates if isinstance(p, str) and p.strip()]
        if not prompts:
            prompts = [prompt]

        total = len(prompts) * batch_size
        tasks = []
        task_idx = 0
        for p_idx, p in enumerate(prompts):
            messages = self._build_messages(p, image_b64_list, system_prompt)
            for b_idx in range(batch_size):
                s = (actual_seed + p_idx * batch_size + b_idx) if seed_mode == "固定种子" else random.randint(0, 2147483647)
                tasks.append((task_idx, self._single_request, (url, api_key, messages, model, aspect_ratio, image_size, s, out_request_id)))
                task_idx += 1

        logger.info(
            "[知衣图生图] 并发发送 %s 个请求（%s 条提示词 × batch_size %s）",
            total,
            len(prompts),
            batch_size,
        )
        results = self._run_concurrent(tasks, label="请求")

        successful = [r for r in results if r is not None]
        if not successful:
            raise RuntimeError("所有请求均失败，无图片返回")

        logger.info("[知衣图生图] 成功 %s/%s", len(successful), total)
        return (torch.cat(successful, dim=0), actual_seed)
