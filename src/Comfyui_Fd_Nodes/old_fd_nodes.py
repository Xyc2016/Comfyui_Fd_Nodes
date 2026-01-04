import json
import os
import time

import oss2
import requests

from .utils.common_util import bytes_calculate_hex_md5, pil2iobyte, tensor2pil
from .config import (
    FD_DOUBAO_KEY,
    FD_DOUBAO_URL,
    FD_OSS_ACCESS_KEY_ID,
    FD_OSS_ACCESS_KEY_SECRET,
    FD_OSS_BUCKET_NAME,
    FD_OSS_ENDPOINT,
    FD_OSS_URL_PATH_PREFIX,
    FD_OSS_URL_PREFIX,
)


class FD_Upload:
    # 文件上传节点
    def __init__(self):
        auth = oss2.Auth(FD_OSS_ACCESS_KEY_ID, FD_OSS_ACCESS_KEY_SECRET)
        self.bucket = oss2.Bucket(
            auth=auth,
            bucket_name=FD_OSS_BUCKET_NAME,
            endpoint=FD_OSS_ENDPOINT,
            connect_timeout=30
        )
        self.oss_url_prefix = FD_OSS_URL_PREFIX

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
            self.bucket.put_object(file_oss_path, file)
            print(f"upload {file_oss_path}")
            return (f"{self.oss_url_prefix}{file_oss_path}",)
        except Exception as e:
            print(e)
            print("上传错误请看上面提示错误")
            return ("",)


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
        raw_data = requests.post(FD_DOUBAO_URL, headers=headers,
                                 data=json.dumps(req_params))

        try:
            data = json.loads(raw_data.content)
            print(data)
            if data["status"] is not True:
                print("上传错误请看上面提示错误或者关闭vpn")
                return (defaultPrompt,)
            else:
                return (data["response"]["Result"],)
        except Exception as e:
            print(e)
            return (defaultPrompt,)