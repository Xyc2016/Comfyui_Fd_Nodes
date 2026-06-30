class ZhiYiImageComboNode:
    """知衣-图片组合节点 - 将最多10张图片+1条提示词打包为组合"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
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
                "prompt_list": ("LIST",),
            },
        }

    RETURN_TYPES = ("ZHIYI_COMBO",)
    RETURN_NAMES = ("combo",)
    FUNCTION = "pack"
    CATEGORY = "知衣/工具"
    OUTPUT_NODE = False

    def pack(self, image_1, prompt, image_2=None, image_3=None,
             image_4=None, image_5=None, image_6=None,
             image_7=None, image_8=None, image_9=None, image_10=None,
             prompt_list=None):
        images = [t for t in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10] if t is not None]
        prompts = []
        if prompt_list and isinstance(prompt_list, list):
            prompts = [p for p in prompt_list if isinstance(p, str) and p.strip()]
        if not prompts:
            prompts = [prompt]
        return ({"images": images, "prompt": prompt, "prompts": prompts},)
