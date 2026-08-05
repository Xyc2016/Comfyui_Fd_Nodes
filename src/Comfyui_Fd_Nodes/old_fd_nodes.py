import json
import re
import time

import requests

from .utils.common_util import bytes_calculate_hex_md5, pil2iobyte, tensor2pil
from .config import (
    FD_DOUBAO_KEY,
    FD_DOUBAO_URL,
    FD_OSS_URL_PATH_PREFIX,
)
from .utils.oss_client import upload_bytes_to_oss


class FD_Upload:
    # 文件上传节点
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "image/upload"

    def gen(self, image):
        file = pil2iobyte(tensor2pil(image))
        if file is None:
            print('bytes_upload_file:: image_bytes is None')
            return ("",)
        try:
            file_oss_path = f"{FD_OSS_URL_PATH_PREFIX}/{bytes_calculate_hex_md5(file)}"
            file_url = upload_bytes_to_oss(file_oss_path, file)
            print(f"upload {file_oss_path}")
            return (file_url,)
        except Exception as e:
            print(e)
            print("上传错误请看上面提示错误")
            return ("",)


def normalize_binary_decision(value, default_value=0):
    """将 VLM 检测输出归一化为 0 或 1 的 int。

    接受 None、空字符串、空白、Markdown code fence、首尾换行及带单一明确
    "0"/"1" 的解释文本；从 "10"、"01"、"1.0" 或同时包含 0 与 1 的歧义文本
    中不提取结果，无法确定时返回 default_value（调用方保证其为 0 或 1）。
    """
    if default_value not in (0, 1):
        default_value = 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else default_value
    if isinstance(value, float):
        return int(value) if value in (0.0, 1.0) else default_value
    if not isinstance(value, str):
        return default_value
    text = value.strip()
    if not text:
        return default_value
    text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if text == "0":
        return 0
    if text == "1":
        return 1
    if text.isdigit():
        return default_value
    tokens = set(re.findall(r"\b0\b|\b1\b", text))
    if tokens == {"0"}:
        return 0
    if tokens == {"1"}:
        return 1
    return default_value


class FD_BinaryDecisionNormalizer:
    # 将文本二值判定结果归一化为 0 或 1 的 int，任何情况下不向后续 switch 传 None
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "default_value": ("INT", {"default": 0, "min": 0, "max": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "normalize"
    OUTPUT_NODE = False
    CATEGORY = "image/captioning"

    def normalize(self, text, default_value=0):
        if default_value not in (0, 1):
            default_value = 0
        result = normalize_binary_decision(text, default_value)
        return (0 if result != 1 else 1,)


class FD_imgToText_Doubao:
    # 调用豆包的图生文节点
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_url": ("STRING", {"multiline": True, "default": "", "forceInput": True},),
                # forceInput 让节点直接显示在连接处
                "prompt": ("STRING", {"default": 'Describe the image below', "multiline": True}),
                "defaultPrompt": ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "image/captioning"

    def gen(self, image_url, prompt, defaultPrompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + FD_DOUBAO_KEY,
        }
        req_params = {
            "text_prompt": prompt
            , "image_url": image_url
            , "response_format": {"type": "text"}
            , "TagTime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            , "extra_body": {"thinking":{"type":"disabled"}}
        }
        try:
            raw_data = requests.post(FD_DOUBAO_URL, headers=headers,
                                     data=json.dumps(req_params))
            data = json.loads(raw_data.content)
            print(data)
            if data["status"] is not True:
                print("上传错误请看上面提示错误或者关闭vpn")
                return (defaultPrompt,)
            return (data["response"]["Result"],)
        except Exception as e:
            print(e)
            return (defaultPrompt,)
