import json
import logging
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Optional

import numpy as np
import oss2
import requests
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from PIL import Image
from pydantic import BaseModel, Field

from .config import (
    FD_GEMINI_URL,
    FD_GEMINI_WEBHOOK_URL,
    FD_OSS_ACCESS_KEY_ID,
    FD_OSS_ACCESS_KEY_SECRET,
    FD_OSS_BUCKET_NAME,
    FD_OSS_ENDPOINT,
    FD_OSS_URL_PATH_PREFIX_GEMINI,
    FD_OSS_URL_PREFIX,
)
from .utils.common_util import (
    bytes_calculate_hex_md5,
    bytesio_to_image_tensor,
    downscale_image_tensor,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GeminiMimeType(str, Enum):
    application_pdf = 'application/pdf'
    audio_mpeg = 'audio/mpeg'
    audio_mp3 = 'audio/mp3'
    audio_wav = 'audio/wav'
    image_png = 'image/png'
    image_jpeg = 'image/jpeg'
    image_webp = 'image/webp'
    text_plain = 'text/plain'
    video_mov = 'video/mov'
    video_mpeg = 'video/mpeg'
    video_mp4 = 'video/mp4'
    video_mpg = 'video/mpg'
    video_avi = 'video/avi'
    video_wmv = 'video/wmv'
    video_mpegps = 'video/mpegps'
    video_flv = 'video/flv'

class GeminiInlineData(BaseModel):
    data: Optional[str] = Field(
        None,
        description='The base64 encoding of the image, PDF, or video to include inline in the prompt. When including media inline, you must also specify the media type (mimeType) of the data. Size limit: 20MB\n',
    )
    mimeType: Optional[GeminiMimeType] = None

class GeminiPart(BaseModel):
    inlineData: Optional[GeminiInlineData] = None
    text: Optional[str] = Field(
        None,
        description='A text prompt or code snippet.',
        examples=['Write a story about a robot learning to paint'],
    )

class GeminiImageModel(str, Enum):
    """
    Gemini Image Model Names allowed by comfy-api
    """

    gemini_2_5_flash_image_preview = "google/gemini-2.5-flash-image-preview"
    gemini_3_pro_image_preview = "google/gemini-3-pro-image-preview"


def fd_gemini_send_webhook(gemini_req_body: dict):
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    headers = {"Content-Type": "application/json"}
    data = {"msgtype": "text", "text": {"content": json.dumps({"datetime": now_str, "gemini_req_body": gemini_req_body}, ensure_ascii=False, indent=4)}}

    requests.post(FD_GEMINI_WEBHOOK_URL, headers=headers, json=data)


class FD_GeminiImage(ComfyNodeABC):
    """
    Node to generate text and image responses from a Gemini model.

    This node allows users to interact with Google's Gemini AI models, providing
    multimodal inputs (text, images, files) to generate coherent
    text and image responses. The node works with the latest Gemini models, handling the
    API communication and response parsing.
    """
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
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "out_request_id": (
                    IO.STRING,
                    {
                        "default": "default",
                        "tooltip": "FD out_request_id for generation",
                    },
                ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for generation",
                    },
                ),
                "model": (
                    IO.COMBO,
                    {
                        "tooltip": "The Gemini model to use for generating responses.",
                        "options": [model.value for model in GeminiImageModel],
                        "default": GeminiImageModel.gemini_2_5_flash_image_preview,
                    },
                ),
                "resolution": (
                    IO.COMBO,
                    {
                        "tooltip": "The Pixel to output for gemini-3-pro-image-preview",
                        "options": ["1K", "2K", "4K"],
                        "default": "2K",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "When seed is fixed to a specific value, the model makes a best effort to provide the same response for repeated requests. Deterministic output isn't guaranteed. Also, changing the model or parameter settings, such as the temperature, can cause variations in the response even when you use the same seed value. By default, a random seed value is used.",
                    },
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional image(s) to use as context for the model. To include multiple images, you can use the Batch Images node.",
                    },
                ),
                "files": (
                    "GEMINI_INPUT_FILES",
                    {
                        "default": None,
                        "tooltip": "Optional file(s) to use as context for the model. Accepts inputs from the Gemini Generate Content Input Files node.",
                    },
                ),
                "aspect_ratio": (
                    IO.COMBO,
                    {
                        "default": "",
                        "options": ["", "1:1", "3:4","9:16"],
                        "tooltip": "Optional aspect ratio for the generated image",
                    }
                ),
                # TODO: later we can add this parameter later
                # "n": (
                #     IO.INT,
                #     {
                #         "default": 1,
                #         "min": 1,
                #         "max": 8,
                #         "step": 1,
                #         "display": "number",
                #         "tooltip": "How many images to generate",
                #     },
                # ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE, IO.STRING)
    FUNCTION = "api_call"
    CATEGORY = "image/generation"
    DESCRIPTION = "Edit images synchronously via Google API."
    API_NODE = True
    GEMINI_URL = FD_GEMINI_URL

    def api_call(self,
        out_request_id: str,
        prompt: str,
        model: str,
        resolution: Optional[str] = None,
        images: Optional[IO.IMAGE] = None,
        aspect_ratio: str = "",
        files: Optional[list[GeminiPart]] = None,
        n=1,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        body = {
            "out_request_id": out_request_id,
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
        }
        if resolution:
            body["resolution"] = resolution
        if images is not None:
            batch_size = images.shape[0]
            image_url_list = []
            for i in range(batch_size):
                single_image = images[i : i + 1]
                scaled_image = downscale_image_tensor(single_image).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_GEMINI}/{bytes_calculate_hex_md5(img_byte_arr)}.png"
                self.bucket.put_object(file_oss_path, img_byte_arr)
                print(f"upload {file_oss_path}")
                oss_file_url = f"{self.oss_url_prefix}{file_oss_path}"
                image_url_list.append(oss_file_url)
            body['image_url_list'] = image_url_list


        if FD_GEMINI_WEBHOOK_URL:
            try:
                print("Sending gemini webhook message...")
                fd_gemini_send_webhook(body)
            except Exception:
                pass

        logger.info(f"Calling Gemini API with {body}")

        response = requests.post(self.GEMINI_URL, json=body)
        if response.status_code != 200:
            raise Exception(f"Failed to call API: {response.content}")
        result = response.json()
        logger.info(f"Gemini API response: {result}")
        result_url = result["result_image_url"]
        image_content = requests.get(result_url).content
        image_bytesio = BytesIO(image_content)
        output_image = bytesio_to_image_tensor(image_bytesio)
        output_text = result["message"]
        return (output_image, output_text,)


