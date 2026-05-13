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

class ZhiYiImageToImageComboNode:
    """知衣图生图节点 - 接收最多8个图片组合并发调用 API"""

    MODELS = [
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image-preview",
        "gemini-3-pro-image-preview-official",
    ]

    ASPECT_RATIOS = ["", "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]
    IMAGE_SIZES = ["4K", "2K", "1080P", "720P"]
    SEED_MODES = ["随机种子", "固定种子"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": ""}),
                "image_size": (cls.IMAGE_SIZES, {"default": "4K"}),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "max_concurrency": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "同时发送请求的最大数量，避免触发API限流",
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
                "out_request_id": ("STRING", {
                    "default": "",
                    "tooltip": "FD out_request_id，留空不传",
                }),
                "combo_1": ("ZHIYI_COMBO",),
                "combo_2": ("ZHIYI_COMBO",),
                "combo_3": ("ZHIYI_COMBO",),
                "combo_4": ("ZHIYI_COMBO",),
                "combo_5": ("ZHIYI_COMBO",),
                "combo_6": ("ZHIYI_COMBO",),
                "combo_7": ("ZHIYI_COMBO",),
                "combo_8": ("ZHIYI_COMBO",),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "log")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate"
    CATEGORY = "知衣/图生图"
    OUTPUT_NODE = False

    def _tensor_to_base64(self, image_tensor):
        """将单帧 tensor (H,W,3) 转为 base64 PNG"""
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _expand_images(self, combo_images):
        """展开 batch tensor 为单帧列表"""
        result = []
        for t in combo_images:
            if t is None:
                continue
            if t.ndim == 4 and t.shape[0] > 1:
                for i in range(t.shape[0]):
                    result.append(t[i:i+1])
            else:
                result.append(t)
        return result

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

    def _single_request(self, url, api_key, messages, model, aspect_ratio, image_size, seed, out_request_id=""):
        payload = {
            "stream": "false",
            "model": model,
            "messages": messages,
            "imageConfig": {"image_size": image_size},
            "modalities": ["image"],
        }
        if aspect_ratio != "auto":
            payload["imageConfig"]["aspect_ratio"] = aspect_ratio
        if out_request_id:
            payload["user"] = out_request_id
        if model not in self.MODELS_NO_SEED:
            payload["seed"] = seed

        log_payload = {
            "url": url,
            "stream": payload["stream"],
            "model": payload["model"],
            "imageConfig": payload["imageConfig"],
            "modalities": payload["modalities"],
            "user": payload.get("user"),
            "seed": payload.get("seed"),
            "messages": self._summarize_messages_for_log(messages),
        }
        print(
            "[知衣图生图-combo] 调用 API 参数: "
            f"{json.dumps(log_payload, ensure_ascii=False)}"
        )
        logger.info("Calling ZhiYi image-to-image combo API with payload=%s", log_payload)
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

    def _run_concurrent(self, tasks, max_workers, label="任务"):
        """tasks: list of (idx, callable, args)，并发执行，返回 (results_list, log_lines)"""
        results = [None] * len(tasks)
        log_lines = []
        workers = min(len(tasks), max(1, max_workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fn, *args): idx
                for idx, fn, args in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    log_lines.append(f"[{label} {idx + 1}] 成功")
                except Exception as e:
                    msg = f"[{label} {idx + 1}] 失败: {type(e).__name__}: {e}"
                    log_lines.append(msg)
                    print(f"[知衣图生图] {msg}")
                    traceback.print_exc()
        return results, log_lines

    def generate(self, model, aspect_ratio, image_size,
                 batch_size=1, max_concurrency=16, seed_mode="随机种子", seed=0,
                 out_request_id="",
                 combo_1=None, combo_2=None, combo_3=None, combo_4=None,
                 combo_5=None, combo_6=None, combo_7=None, combo_8=None,
                 system_prompt=""):
        cfg = load_config()
        final_base_url = FD_LITELLM_BASE_URL
        final_api_key = FD_LITELLM_API_KEY
        if not final_base_url or final_base_url == "https://your-api-base-url":
            raise RuntimeError("未配置 base_url，请在节点输入或环境变量 FD_LITELLM_BASE_URL 中设置")
        if not final_api_key or final_api_key == "your-api-key":
            raise RuntimeError("未配置 api_key，请在节点输入或环境变量 FD_LITELLM_API_KEY 中设置")
        url = f"{final_base_url}/v1/chat/completions"

        actual_seed = random.randint(0, 2147483647) if seed_mode == "随机种子" else seed

        combos = [c for c in [combo_1, combo_2, combo_3, combo_4, combo_5, combo_6, combo_7, combo_8] if c is not None]
        if not combos:
            raise RuntimeError("未提供任何组合输入，请连接至少一个 combo")

        tasks = []
        task_idx = 0
        pre_errors = []
        for c_idx, combo in enumerate(combos):
            try:
                if not isinstance(combo, dict):
                    raise RuntimeError(f"数据格式错误: 期望 dict，实际 {type(combo).__name__}")
                combo_images = combo.get("images", [])
                combo_prompts = combo.get("prompts") or [combo.get("prompt", "")]
                if not combo_images:
                    raise RuntimeError("无图片")

                expanded_images = self._expand_images(combo_images)
                image_b64_list = [self._tensor_to_base64(t) for t in expanded_images]
                for p_idx, p in enumerate(combo_prompts):
                    messages = self._build_messages(p, image_b64_list, system_prompt)
                    for b_idx in range(batch_size):
                        s = (actual_seed + task_idx) if seed_mode == "固定种子" else random.randint(0, 2147483647)
                        tasks.append((task_idx, self._single_request, (url, final_api_key, messages, model, aspect_ratio or None, image_size, s, out_request_id)))
                        task_idx += 1
            except Exception as e:
                msg = f"[combo_{c_idx + 1}] 预处理失败，跳过: {type(e).__name__}: {e}"
                pre_errors.append(msg)
                print(f"[知衣图生图] {msg}")
                traceback.print_exc()

        if not tasks:
            err = "\n".join(pre_errors) if pre_errors else "所有组合均无有效图片"
            raise RuntimeError(f"无可发送请求\n{err}")

        print(f"[知衣图生图] 并发发送 {len(tasks)} 个请求（{len(combos)} 个组合 × batch_size {batch_size}），并发上限 {max_concurrency}")
        results, log_lines = self._run_concurrent(tasks, max_concurrency, label="请求")

        successful = [r for r in results if r is not None]
        header = f"总计: {len(successful)}/{len(tasks)} 成功, url={url}"
        if pre_errors:
            header += f"\n预处理失败 {len(pre_errors)} 个: " + "; ".join(pre_errors)
        log_lines.insert(0, header)
        log_text = "\n".join(log_lines)

        if not successful:
            raise RuntimeError(f"所有请求均失败，无图片返回\n{log_text}")

        print(f"[知衣图生图] 成功 {len(successful)}/{len(tasks)}")
        return (successful, actual_seed, log_text)
