import requests
import json
import logging
from .config_manager import load_config
from .utils.logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)

class ZhiYiTextGenNode:
    """知依文生文节点 - 调用 Gemini 模型生成文本"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "请介绍你自己",
                    "multiline": True,
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
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "知衣/文生文"
    OUTPUT_NODE = False

    def _summarize_response_for_log(self, result):
        summary = {"keys": list(result.keys())}
        choices = result.get("choices", [])
        summary["choice_count"] = len(choices)
        if not choices:
            return summary

        message = choices[0].get("message", {})
        summary["message_keys"] = list(message.keys())
        content = message.get("content", [])
        if isinstance(content, list):
            summary["text_part_count"] = sum(
                1 for part in content if isinstance(part, dict) and part.get("type") == "text"
            )
        elif isinstance(content, str):
            summary["content_length"] = len(content)
        return summary

    def _summarize_messages_for_log(self, messages):
        summarized_messages = []
        for message in messages:
            content = message.get("content", "")
            summarized_message = {"role": message.get("role", "")}
            if isinstance(content, str):
                summarized_message["text"] = content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                summarized_message["text"] = "\n".join(text_parts)
            else:
                summarized_message["content"] = content
            summarized_messages.append(summarized_message)
        return summarized_messages

    def generate(self, prompt, node_switch=1,
                 system_prompt="", temperature=0.7, max_tokens=2048):
        if node_switch == 1:
            return ("",)

        cfg = load_config()
        base_url = cfg["base_url"]
        api_key = cfg["api_key"]
        base_url = base_url.rstrip("/")
        url = f"{base_url}/v1/chat/completions"

        messages = []
        if system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        messages.append({
            "role": "user",
            "content": prompt,
        })

        payload = {
            "stream": False,
            "model": "doubao-seed-2.0-mini",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.info(
            "Calling ZhiYi text API with payload=%s",
            {
                "url": url,
                "stream": payload["stream"],
                "model": payload["model"],
                "temperature": payload["temperature"],
                "max_tokens": payload["max_tokens"],
                "messages": self._summarize_messages_for_log(messages),
            },
        )

        try:
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
                "ZhiYi text API response summary: %s",
                {
                    "status_code": response.status_code,
                    **self._summarize_response_for_log(result),
                },
            )
            text = result["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )
            return (text,)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API 请求失败: {e}")
        except RuntimeError:
            raise
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"解析响应失败: {e}\n原始响应: {response.text[:500]}")
