import io
import hashlib
import os
from PIL import Image
import torch
import numpy as np
from io import BytesIO
import math
from comfy.utils import common_upscale


def pil2iobyte(pil_image,format='PNG'):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format=format)  
    byte_arr = byte_arr.getvalue() 
    return byte_arr

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def bytes_calculate_hex_md5(img_bytes, block_size=64 * 1024):
    md5 = hashlib.md5()
    img_arr = [img_bytes[i:i + block_size] for i in range(0, len(img_bytes), block_size)]
    for chunk in img_arr:
        md5.update(chunk)
    return md5.hexdigest()

def bytesio_to_image_tensor(image_bytesio: BytesIO, mode: str = "RGBA") -> torch.Tensor:
    image = Image.open(image_bytesio)
    image = image.convert(mode)
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)

def downscale_image_tensor(image, total_pixels=1536 * 1024) -> torch.Tensor:
    """Downscale input image tensor to roughly the specified total pixels."""
    samples = image.movedim(-1, 1)
    total = int(total_pixels)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1, -1)
    return s

def validate_string(
    string: str,
    strip_whitespace=True,
    field_name="prompt",
    min_length=None,
    max_length=None,
):
    if string is None:
        raise Exception(f"Field '{field_name}' cannot be empty.")
    if strip_whitespace:
        string = string.strip()
    if min_length and len(string) < min_length:
        raise Exception(
            f"Field '{field_name}' cannot be shorter than {min_length} characters; was {len(string)} characters long."
        )
    if max_length and len(string) > max_length:
        raise Exception(
            f" Field '{field_name} cannot be longer than {max_length} characters; was {len(string)} characters long."
        )