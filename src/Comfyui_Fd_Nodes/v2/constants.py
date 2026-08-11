"""V2 节点类别中文翻译表与多选下拉 options。"""

from ..zhiyi_rmbg_segment_node import BODY_CLASSES, CLOTHES_CLASSES, FASHION_CLASSES


CLOTHES_CLASS_LABELS = {
    "Background": "背景",
    "Hat": "帽子",
    "Hair": "头发",
    "Sunglasses": "太阳镜",
    "Upper-clothes": "上衣",
    "Skirt": "裙子",
    "Pants": "裤子",
    "Dress": "连衣裙",
    "Belt": "腰带",
    "Left-shoe": "左脚鞋",
    "Right-shoe": "右脚鞋",
    "Face": "脸",
    "Left-leg": "左腿",
    "Right-leg": "右腿",
    "Left-arm": "左臂",
    "Right-arm": "右臂",
    "Bag": "包",
    "Scarf": "围巾",
}

FASHION_CLASS_LABELS = {
    "unlabelled": "未标注",
    "shirt, blouse": "衬衫/女士衬衫",
    "top, t-shirt, sweatshirt": "上衣/T恤/卫衣",
    "sweater": "毛衣",
    "cardigan": "开衫",
    "jacket": "夹克",
    "vest": "背心",
    "pants": "裤子",
    "shorts": "短裤",
    "skirt": "裙子",
    "coat": "外套",
    "dress": "连衣裙",
    "jumpsuit": "连体裤",
    "cape": "斗篷",
    "glasses": "眼镜",
    "hat": "帽子",
    "headband, head covering, hair accessory": "发带/头饰/发饰",
    "tie": "领带",
    "glove": "手套",
    "watch": "手表",
    "belt": "腰带",
    "leg warmer": "腿套",
    "tights, stockings": "紧身裤/长袜",
    "sock": "袜子",
    "shoe": "鞋",
    "bag, wallet": "包/钱包",
    "scarf": "围巾",
    "umbrella": "雨伞",
    "hood": "兜帽",
    "collar": "领子",
    "lapel": "翻领",
    "epaulette": "肩章",
    "sleeve": "袖子",
    "pocket": "口袋",
    "neckline": "领口",
    "buckle": "搭扣",
    "zipper": "拉链",
    "applique": "贴花",
    "bead": "珠饰",
    "bow": "蝴蝶结",
    "flower": "花朵",
    "fringe": "流苏",
    "ribbon": "缎带",
    "rivet": "铆钉",
    "ruffle": "褶边",
    "sequin": "亮片",
    "tassel": "穗饰",
}

BODY_CLASS_LABELS = {
    "Hair": "头发",
    "Glasses": "眼镜",
    "Top-clothes": "上衣",
    "Bottom-clothes": "下装",
    "Torso-skin": "躯干皮肤",
    "Face": "脸",
    "Left-arm": "左臂",
    "Right-arm": "右臂",
    "Left-leg": "左腿",
    "Right-leg": "右腿",
    "Left-foot": "左脚",
    "Right-foot": "右脚",
}


def _validate_class_tables(classes, labels):
    missing = set(classes) - set(labels)
    extra = set(labels) - set(classes)
    duplicates = [name for name in classes if classes.count(name) > 1]
    blanks = [name for name, label in labels.items() if not label]
    problems = []
    if missing:
        problems.append(f"缺少翻译: {sorted(missing)}")
    if extra:
        problems.append(f"多余翻译: {sorted(extra)}")
    if duplicates:
        problems.append(f"类别重复: {sorted(set(duplicates))}")
    if blanks:
        problems.append(f"空翻译: {sorted(blanks)}")
    if problems:
        raise RuntimeError("类别翻译表不一致: " + "; ".join(problems))


_validate_class_tables(CLOTHES_CLASSES, CLOTHES_CLASS_LABELS)
_validate_class_tables(FASHION_CLASSES, FASHION_CLASS_LABELS)
_validate_class_tables(BODY_CLASSES, BODY_CLASS_LABELS)

# 多选下拉使用纯英文字符串：ComfyUI 0.6.0 后端校验会原样提交对象，
# 对象 options 会导致 "Value not in list" 校验失败。中文标签需前端扩展映射。
CLOTHES_CLASS_OPTIONS = list(CLOTHES_CLASSES)
FASHION_CLASS_OPTIONS = list(FASHION_CLASSES)
BODY_CLASS_OPTIONS = list(BODY_CLASSES)
