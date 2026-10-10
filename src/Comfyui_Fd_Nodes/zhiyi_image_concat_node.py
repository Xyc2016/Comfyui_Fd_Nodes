import numpy as np
import torch
from PIL import Image

# 各背景选项对应的 RGBA 填充色，RGB 模式下取前三通道
BACKGROUND_COLORS = {
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
    "transparent": (0, 0, 0, 0),
}


class ZhiYiImageConcatNode:
    """知衣-组合图像(最长边)节点 - 按最长边拼接最多10张图像，不裁剪不缩放，空缺处填充空白。

    与 ComfyUI 原生 ImageConcatenate 的区别：原生节点在尺寸不一致时会裁剪图像，
    本节点以各方向的最大值为画布尺寸，图像原始内容完整保留。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE", {"tooltip": "第一张图像，其余图像按序号排在其后"}),
                "direction": (["horizontal", "vertical"], {
                    "default": "horizontal",
                    "tooltip": "拼接方向：horizontal = 左右拼接，vertical = 上下拼接",
                }),
                "alignment": (["start", "center", "end"], {
                    "default": "start",
                    "tooltip": "对齐方式（垂直于拼接方向的排布）：水平拼接时 start=顶部对齐 / center=垂直居中 / end=底部对齐；垂直拼接时 start=左对齐 / center=水平居中 / end=右对齐",
                }),
                "background": (["white", "black", "transparent"], {
                    "default": "white",
                    "tooltip": "空缺处填充：white=白色，black=黑色，transparent=透明（输出带 alpha 通道）",
                }),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "concat"
    CATEGORY = "知衣/工具"
    OUTPUT_NODE = False

    def concat(self, image_1, image_2=None, direction="horizontal", alignment="start",
               background="white", image_3=None, image_4=None, image_5=None,
               image_6=None, image_7=None, image_8=None, image_9=None, image_10=None):
        tensors = [t for t in (image_1, image_2, image_3, image_4, image_5,
                               image_6, image_7, image_8, image_9, image_10) if t is not None]
        if len(tensors) < 2:
            raise ValueError("知衣-组合图像(最长边)至少需要连接两张图像")
        batch = max(t.shape[0] for t in tensors)
        results = []
        for i in range(batch):
            pil_images = [self._tensor_to_pil(t[i % t.shape[0]]) for t in tensors]
            results.append(self._pil_to_tensor(self._concat_images(pil_images, direction, alignment, background)))
        return (torch.cat(results, dim=0),)

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        array = (tensor.cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        if array.shape[-1] == 4:
            return Image.fromarray(array, mode="RGBA")
        return Image.fromarray(array)

    @staticmethod
    def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
        array = np.asarray(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _concat_images(self, pil_images, direction, alignment, background):
        mode = "RGB"
        if background == "transparent" or any("A" in img.mode for img in pil_images):
            mode = "RGBA"
        images = [img.convert(mode) for img in pil_images]

        if direction == "horizontal":
            canvas_size = (sum(img.width for img in images), max(img.height for img in images))
        else:
            canvas_size = (max(img.width for img in images), sum(img.height for img in images))

        fill = BACKGROUND_COLORS[background]
        if mode == "RGB":
            fill = fill[:3]
        canvas = Image.new(mode, canvas_size, fill)
        offset_main = 0
        for img in images:
            if direction == "horizontal":
                dest = (offset_main, self._offset(canvas_size[1] - img.height, alignment))
                offset_main += img.width
            else:
                dest = (self._offset(canvas_size[0] - img.width, alignment), offset_main)
                offset_main += img.height
            if mode == "RGBA":
                # alpha_composite 正确处理半透明像素，避免 paste 双重调制 alpha
                canvas.alpha_composite(img, dest=dest)
            else:
                canvas.paste(img, dest)
        return canvas

    @staticmethod
    def _offset(missing: int, alignment: str) -> int:
        if alignment == "center":
            return missing // 2
        if alignment == "end":
            return missing
        return 0
