import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config_manager import load_config
from .utils.error_utils import normalize_error_message
from .utils.logging_utils import configure_default_logging
from .zhiyi_image_text_node import MAX_REQUEST_BODY_BYTES, ZhiYiImageTextNode

configure_default_logging()
logger = logging.getLogger(__name__)


class ZhiYiImageTextComboNode(ZhiYiImageTextNode):
    """知衣多图图生文节点 - 接收最多8个图片组合并发调用豆包视觉理解 API。"""

    DEFAULT_PROMPT = "介绍这幅图片"
    MODEL = "doubao-seed-2.0-mini"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_concurrency": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "同时发送请求的最大数量，避免触发 API 限流",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                }),
                "retry_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 3,
                    "step": 1,
                    "tooltip": "请求失败后的重试次数。仅重试网络错误、超时、429 和 5xx",
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "log")
    FUNCTION = "generate"
    CATEGORY = "知衣/图生文"
    OUTPUT_NODE = False

    def _expand_images(self, combo_images):
        result = []
        for image in combo_images:
            if image is None:
                continue
            if image.ndim == 4:
                for i in range(image.shape[0]):
                    result.append(image[i:i + 1])
            elif image.ndim == 3:
                result.append(image.unsqueeze(0))
            else:
                raise RuntimeError(f"图片 tensor 维度错误: {image.ndim}")
        return result

    def _select_prompt(self, combo):
        combo_prompts = combo.get("prompts")
        if isinstance(combo_prompts, str):
            combo_prompts = [combo_prompts]
        if isinstance(combo_prompts, list):
            for prompt in combo_prompts:
                if isinstance(prompt, str) and prompt.strip():
                    return prompt.strip()

        prompt = combo.get("prompt", "")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
        return self.DEFAULT_PROMPT

    def _build_messages(self, prompt, data_urls, system_prompt=""):
        messages = []
        system_text = (system_prompt or "").strip()
        if system_text:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            })

        content = [{"type": "text", "text": prompt}]
        for data_url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages.append({"role": "user", "content": content})
        return messages

    def _extract_text(self, result):
        text = result["choices"][0]["message"]["content"]
        if isinstance(text, list):
            return "".join(
                part.get("text", "") for part in text if isinstance(part, dict)
            )
        if text is None:
            return ""
        return str(text)

    def _is_retryable_status(self, status_code):
        return status_code == 429 or 500 <= status_code < 600

    def _is_retryable_exception(self, exc):
        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True

        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code is not None and self._is_retryable_status(status_code)

    def _sleep_before_retry(self, attempt_index):
        time.sleep(min(2 ** attempt_index, 2))

    def _single_request(self, url, api_key, request_body, request_log, retry_count):
        max_attempts = max(1, int(retry_count) + 1)
        last_error = None

        for attempt_index in range(max_attempts):
            response = None
            try:
                logger.info(
                    "Calling ZhiYi image-to-text combo API with payload=%s",
                    {
                        **request_log,
                        "attempt": attempt_index + 1,
                        "max_attempts": max_attempts,
                    },
                )
                response = requests.post(
                    url=url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                    data=request_body,
                    timeout=(5, 120),
                )

                status_code = getattr(response, "status_code", None)
                if status_code is not None and status_code >= 400:
                    response_text = getattr(response, "text", "")[:1000]
                    last_error = RuntimeError(f"API 请求失败: {status_code}\n{response_text}")
                    if self._is_retryable_status(status_code) and attempt_index < max_attempts - 1:
                        logger.warning(
                            "ZhiYi image-to-text combo API attempt %s/%s failed with HTTP %s, retrying",
                            attempt_index + 1,
                            max_attempts,
                            status_code,
                        )
                        self._sleep_before_retry(attempt_index)
                        continue
                    raise last_error

                response.raise_for_status()
                result = response.json()
                logger.info(
                    "ZhiYi image-to-text combo API response summary: %s",
                    {
                        "status_code": response.status_code,
                        **self._summarize_response_for_log(result),
                    },
                )
                return self._extract_text(result), attempt_index
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                response_text = response.text[:500] if response is not None else ""
                raise RuntimeError(f"解析响应失败: {exc}\n原始响应: {response_text}") from exc
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if self._is_retryable_exception(exc) and attempt_index < max_attempts - 1:
                    logger.warning(
                        "ZhiYi image-to-text combo API attempt %s/%s failed, retrying: %s",
                        attempt_index + 1,
                        max_attempts,
                        exc,
                    )
                    self._sleep_before_retry(attempt_index)
                    continue
                raise RuntimeError(f"API 请求失败: {exc}") from exc

        raise RuntimeError(last_error or "API 请求失败")

    def _run_concurrent(self, tasks, max_workers, label="请求"):
        results = {}
        log_lines = []
        last_error_message = None
        workers = min(len(tasks), max(1, int(max_workers)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fn, *args): (result_index, combo_slot)
                for result_index, combo_slot, fn, args in tasks
            }
            for future in as_completed(futures):
                result_index, combo_slot = futures[future]
                try:
                    text, retry_used = future.result()
                    results[result_index] = {"ok": True, "text": text}
                    suffix = f"，重试 {retry_used} 次" if retry_used else ""
                    log_lines.append(f"[{label} combo_{combo_slot}] 成功{suffix}")
                except Exception as exc:
                    normalized_error = normalize_error_message(exc)
                    last_error_message = normalized_error
                    results[result_index] = {"ok": False, "text": f"ERROR: {normalized_error}"}
                    log_lines.append(f"[{label} combo_{combo_slot}] 失败: {normalized_error}")
                    print(f"[知衣图生文-combo] combo_{combo_slot} 失败: {normalized_error}")
                    traceback.print_exc()
        return results, log_lines, last_error_message

    def _format_text_output(self, combo_inputs, result_items):
        successful = [item for item in result_items if item and item.get("ok")]
        if len(combo_inputs) == 1 and len(successful) == 1:
            return successful[0]["text"]

        sections = []
        for index, (combo_slot, _combo) in enumerate(combo_inputs):
            item = result_items[index] or {
                "ok": False,
                "text": "ERROR: UNKNOWN: 未执行",
            }
            sections.append(f"[combo_{combo_slot}]\n{item['text']}")
        return "\n\n".join(sections)

    def generate(
        self,
        max_concurrency=8,
        temperature=0.7,
        max_tokens=2048,
        retry_count=1,
        node_switch=0,
        combo_1=None,
        combo_2=None,
        combo_3=None,
        combo_4=None,
        combo_5=None,
        combo_6=None,
        combo_7=None,
        combo_8=None,
        system_prompt="",
    ):
        if node_switch == 1:
            return ("", "node skipped")

        cfg = load_config()
        base_url = cfg["base_url"].rstrip("/")
        api_key = cfg["api_key"]
        url = f"{base_url}/v1/chat/completions"

        combo_inputs = [
            (slot, combo)
            for slot, combo in enumerate(
                [combo_1, combo_2, combo_3, combo_4, combo_5, combo_6, combo_7, combo_8],
                start=1,
            )
            if combo is not None
        ]
        if not combo_inputs:
            raise RuntimeError("未提供任何组合输入，请连接至少一个 combo")

        result_items = [None] * len(combo_inputs)
        tasks = []
        pre_errors = []
        last_error_message = None

        for result_index, (combo_slot, combo) in enumerate(combo_inputs):
            try:
                if not isinstance(combo, dict):
                    raise RuntimeError(f"数据格式错误: 期望 dict，实际 {type(combo).__name__}")

                combo_images = combo.get("images", [])
                if not combo_images:
                    raise RuntimeError("无图片")

                expanded_images = self._expand_images(combo_images)
                if not expanded_images:
                    raise RuntimeError("无有效图片")

                prompt = self._select_prompt(combo)
                placeholder_messages = self._build_messages(
                    prompt, [""] * len(expanded_images), system_prompt
                )
                placeholder_payload = self._build_request_payload(
                    placeholder_messages, temperature, max_tokens
                )
                image_budget = self._image_budget(
                    placeholder_payload, len(expanded_images)
                )

                data_urls = []
                image_infos = []
                for image in expanded_images:
                    data_url, image_info = self._image_tensor_to_data_url(
                        image, max_data_url_bytes=image_budget
                    )
                    data_urls.append(data_url)
                    image_infos.append(image_info)

                messages = self._build_messages(prompt, data_urls, system_prompt)
                payload = self._build_request_payload(messages, temperature, max_tokens)
                request_body, request_body_bytes = self._serialize_request_body(payload)
                for image_index, image_info in enumerate(image_infos, start=1):
                    image_info["combo"] = combo_slot
                    image_info["image"] = image_index
                    image_info["request_body_bytes"] = request_body_bytes
                    image_info["max_request_body_bytes"] = MAX_REQUEST_BODY_BYTES
                    logger.info(
                        "ZhiYi image-to-text combo encoded image: %s",
                        image_info,
                    )

                request_log = {
                    "url": url,
                    "stream": payload["stream"],
                    "model": payload["model"],
                    "temperature": payload["temperature"],
                    "max_tokens": payload["max_tokens"],
                    "request_body_bytes": request_body_bytes,
                    "max_request_body_bytes": MAX_REQUEST_BODY_BYTES,
                    "messages": self._summarize_messages_for_log(messages),
                }
                tasks.append(
                    (
                        result_index,
                        combo_slot,
                        self._single_request,
                        (
                            url,
                            api_key,
                            request_body,
                            request_log,
                            retry_count,
                        ),
                    )
                )
            except Exception as exc:
                normalized_error = normalize_error_message(exc)
                last_error_message = normalized_error
                result_items[result_index] = {"ok": False, "text": f"ERROR: {normalized_error}"}
                msg = f"[combo_{combo_slot}] 预处理失败: {normalized_error}"
                pre_errors.append(msg)
                print(f"[知衣图生文-combo] {msg}")
                traceback.print_exc()

        run_results = {}
        run_log_lines = []
        if tasks:
            print(
                f"[知衣图生文-combo] 并发发送 {len(tasks)} 个请求（{len(combo_inputs)} 个组合），"
                f"并发上限 {max_concurrency}"
            )
            run_results, run_log_lines, run_last_error = self._run_concurrent(
                tasks,
                max_concurrency,
                label="请求",
            )
            if run_last_error:
                last_error_message = run_last_error

        for result_index, item in run_results.items():
            result_items[result_index] = item

        successful_count = sum(1 for item in result_items if item and item.get("ok"))
        header = (
            f"总计: {successful_count}/{len(combo_inputs)} 成功, "
            f"url={url}, 并发上限={max_concurrency}, retry_count={retry_count}"
        )
        log_lines = [header]
        if pre_errors:
            log_lines.extend(pre_errors)
        log_lines.extend(run_log_lines)
        log_text = "\n".join(log_lines)

        if successful_count == 0:
            raise RuntimeError(
                last_error_message or normalize_error_message(f"所有请求均失败，无文本返回\n{log_text}")
            )

        return (self._format_text_output(combo_inputs, result_items), log_text)
