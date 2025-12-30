import io
import os
from inspect import cleandoc
from typing import Any, Dict, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from uuid import uuid4

FD_REMOVE_WATERMARK_SERVICE_URL = os.getenv("FD_REMOVE_WATERMARK_SERVICE_URL", "http://localhost:8000/v1/process")


class FD_RemoveWatermark:
    """
    Remove watermark from image using AI inpainting service
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {"default": FD_REMOVE_WATERMARK_SERVICE_URL, "multiline": False}),
                "text_prompt": (
                    "STRING",
                    {"default": "sticker. label. badge. corner tag. overlay logo.", "multiline": True},
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.30,
                    },
                ),
                "text_threshold": (
                    "FLOAT",
                    {
                        "default": 0.25,
                    },
                ),
                "max_side": (
                    "INT",
                    {
                        "default": 3000,
                    },
                ),
                "mask_dilate_ksize": (
                    "INT",
                    {
                        "default": 5,
                    },
                ),
                "mask_dilate_iters": (
                    "INT",
                    {
                        "default": 1,
                    },
                ),
                "jpeg_quality": (
                    "INT",
                    {
                        "default": 95,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_watermark"
    CATEGORY = "image/postprocessing"

    def remove_watermark(
        self,
        image: torch.Tensor,
        service_url: str,
        text_prompt: str,
        threshold: float,
        text_threshold: float,
        max_side: int,
        mask_dilate_ksize: int,
        mask_dilate_iters: int,
        jpeg_quality: int,
    ) -> Tuple[torch.Tensor]:
        call_id = str(uuid4())[:8]
        print(
            f"FD_RemoveWatermark [{call_id}] Start processing. ",
            {
                "call_id": call_id,
                "image_shape": list(image.shape),
                "service_url": service_url,
                "text_prompt": text_prompt,
                "threshold": threshold,
                "text_threshold": text_threshold,
                "max_side": max_side,
                "mask_dilate_ksize": mask_dilate_ksize,
                "mask_dilate_iters": mask_dilate_iters,
                "jpeg_quality": jpeg_quality,
            },
        )
        try:
            # Convert ComfyUI tensor to PIL Image (only process first image in batch)
            img_tensor = image[0]  # [H, W, C] in range [0, 1]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Convert to bytes
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Prepare request
            files = {"image": ("image.png", img_bytes, "image/png")}
            data = {
                "text_prompt": text_prompt,
                "threshold": threshold,
                "text_threshold": text_threshold,
                "max_side": max_side,
                "mask_dilate_ksize": mask_dilate_ksize,
                "mask_dilate_iters": mask_dilate_iters,
                "inpaint_method": "lama",
                "enable_quality_check": True,
                "fallback_to_original": True,
                "jpeg_quality": jpeg_quality,
            }

            print(f"FD_RemoveWatermark [{call_id}] Calling service at {service_url} ...")
            # Call service
            response = requests.post(service_url, files=files, data=data, timeout=60)
            response.raise_for_status()

            # Convert response back to ComfyUI tensor
            result_img = Image.open(io.BytesIO(response.content)).convert("RGB")
            result_np = np.array(result_img).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_np).unsqueeze(0)  # [1, H, W, C]

            print(f"FD_RemoveWatermark [{call_id}] Successfully processed image.")
            return (result_tensor,)

        except Exception as e:
            # Return original image on any error
            print(f"FD_RemoveWatermark [{call_id}] error: {e}, returning original image")
            return (image,)


class Example:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("Image", { "tooltip": "This is an image"}),
                "int_field": ("INT", {
                    "default": 0,
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "float_field": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "test"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Example"

    def test(self, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"FD_RemoveWatermark": FD_RemoveWatermark}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"FD_RemoveWatermark": "Remove Watermark"}
