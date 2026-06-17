import logging
from typing import Optional

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict

from .config import FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL
from .utils.common_util import bytesio_to_image_tensor
from .utils.gpt_image_edit_client import get_default_gpt_image_edit_client
from .utils.gpt_image_size import resolution_to_image_generation_edit_size
from .utils.logging_utils import configure_default_logging
from .utils.webhook import webhook_send

configure_default_logging()
logger = logging.getLogger(__name__)


def _resolution_to_edit_size(resolution: str, aspect_ratio: str) -> str:
    return resolution_to_image_generation_edit_size(resolution, aspect_ratio)


class FD_GTPImage(ComfyNodeABC):
    """
    Node to edit images using GPT Image API.
    """

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
                        "tooltip": "The GPT image model to use for image edits.",
                        "options": ["gpt-image-2"],
                        "default": "gpt-image-2",
                    },
                ),
                "resolution": (
                    IO.COMBO,
                    {
                        "tooltip": "Output size preset for the image edit request.",
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
                        "tooltip": "Reserved for compatibility with FD_GeminiImage input format.",
                    },
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Input image(s) to edit. Multiple images are sent as repeated multipart fields.",
                    },
                ),
                "files": (
                    "GEMINI_INPUT_FILES",
                    {
                        "default": None,
                        "tooltip": "Reserved for compatibility with FD_GeminiImage input format.",
                    },
                ),
                "aspect_ratio": (
                    IO.COMBO,
                    {
                        "default": "",
                        "options": ["", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9"],
                        "tooltip": "Optional aspect ratio for the edited image.",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "default": "medium",
                        "options": ["low", "medium", "high"],
                        "tooltip": "输出图片质量，默认 medium。追加到 optional 末尾以兼容旧 workflow 的 widget 顺序。",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE, IO.STRING, IO.STRING)
    FUNCTION = "api_call"
    CATEGORY = "image/generation"
    DESCRIPTION = "Edit images synchronously via GPT Image API."
    API_NODE = True

    def api_call(
        self,
        out_request_id: str,
        prompt: str,
        model: str,
        resolution: Optional[str] = None,
        quality: str = "medium",
        images: Optional[IO.IMAGE] = None,
        aspect_ratio: str = "",
        files=None,
        seed: int = 42,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        del files, seed, unique_id, kwargs

        if images is None:
            raise ValueError("FD_GTPImage requires at least one input image.")
        if not prompt or not prompt.strip():
            raise ValueError("FD_GTPImage requires a non-empty prompt.")

        size = _resolution_to_edit_size(resolution or "2K", aspect_ratio)
        request_images = [images[i : i + 1] for i in range(images.shape[0])]

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending gtp_image webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "gtp_image_request": {
                        "data": {
                            "model": model,
                            "prompt": prompt.strip(),
                            "size": size,
                            "quality": quality,
                        },
                        "image_count": len(request_images),
                    }
                })
            except Exception:
                pass

        logger.info(
            "Calling GPT Image edit with model=%s size=%s quality=%s image_count=%s",
            model,
            size,
            quality,
            len(request_images),
        )

        client = get_default_gpt_image_edit_client()
        image_bytesio, output_text, result_url = client.edit_image(
            image_tensors=request_images,
            prompt=prompt.strip(),
            size=size,
            quality=quality,
            out_request_id=out_request_id if out_request_id != "default" else "",
        )
        output_image = bytesio_to_image_tensor(image_bytesio)
        return (output_image, output_text, result_url)
