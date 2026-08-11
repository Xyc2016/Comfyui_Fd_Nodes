import base64
import io
import json

import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import sam2_segment_node as sam2_module
from src.Comfyui_Fd_Nodes import zhiyi_controlnet_aux_node as aux_module
from src.Comfyui_Fd_Nodes import zhiyi_dwpose_detect_node as dwpose_module
from src.Comfyui_Fd_Nodes import zhiyi_rmbg_segment_node as rmbg_module
from src.Comfyui_Fd_Nodes.config import (
    DWPOSE_SERVICE_ENVS,
    RMBG_SERVICE_ENVS,
    SAM2_SERVICE_ENVS,
)
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.v2 import controlnet_aux_v2_node as aux_v2_module
from src.Comfyui_Fd_Nodes.v2.constants import (
    BODY_CLASS_OPTIONS,
    CLOTHES_CLASS_OPTIONS,
    FASHION_CLASS_OPTIONS,
    _build_class_options,
)
from src.Comfyui_Fd_Nodes.v2.controlnet_aux_v2_node import (
    ZhiYiDepthAnythingV2PreprocessorNodeV2,
    ZhiYiDWPoseDetectNodeV2,
    ZhiYiLineArtPreprocessorNodeV2,
)
from src.Comfyui_Fd_Nodes.v2.env import (
    ENV_OPTIONS,
    CUSTOM_ENV,
    resolve_service_env,
)
from src.Comfyui_Fd_Nodes.v2.rmbg_segment_v2_node import (
    ZhiYiBodySegmentNodeV2,
    ZhiYiClothesSegmentNodeV2,
    ZhiYiFashionSegmentNodeV2,
    ZhiYiRMBGNodeV2,
)
from src.Comfyui_Fd_Nodes.v2.sam2_segment_v2_node import ZhiYiSAM2SegmentNodeV2
from src.Comfyui_Fd_Nodes.zhiyi_rmbg_segment_node import (
    BODY_CLASSES,
    CLOTHES_CLASSES,
    FASHION_CLASSES,
    ZhiYiBodySegmentNode,
    ZhiYiClothesSegmentNode,
    ZhiYiFashionSegmentNode,
    ZhiYiRMBGNode,
)
from src.Comfyui_Fd_Nodes.zhiyi_controlnet_aux_node import (
    ZhiYiDepthAnythingV2PreprocessorNode,
    ZhiYiLineArtPreprocessorNode,
)
from src.Comfyui_Fd_Nodes.zhiyi_dwpose_detect_node import ZhiYiDWPoseDetectNode
from src.Comfyui_Fd_Nodes.sam2_segment_node import ZhiYiSAM2SegmentNode

V2_CLASSES = [
    ZhiYiRMBGNodeV2,
    ZhiYiClothesSegmentNodeV2,
    ZhiYiFashionSegmentNodeV2,
    ZhiYiBodySegmentNodeV2,
    ZhiYiSAM2SegmentNodeV2,
    ZhiYiDWPoseDetectNodeV2,
    ZhiYiLineArtPreprocessorNodeV2,
    ZhiYiDepthAnythingV2PreprocessorNodeV2,
]

OLD_CLASSES = [
    ZhiYiRMBGNode,
    ZhiYiClothesSegmentNode,
    ZhiYiFashionSegmentNode,
    ZhiYiBodySegmentNode,
    ZhiYiSAM2SegmentNode,
    ZhiYiDWPoseDetectNode,
    ZhiYiLineArtPreprocessorNode,
    ZhiYiDepthAnythingV2PreprocessorNode,
]

MAPPING_KEYS = [
    "ZhiYiRMBGNodeV2",
    "ZhiYiClothesSegmentNodeV2",
    "ZhiYiFashionSegmentNodeV2",
    "ZhiYiBodySegmentNodeV2",
    "ZhiYiSAM2SegmentNodeV2",
    "ZhiYiDWPoseDetectNodeV2",
    "ZhiYiLineArtPreprocessorNodeV2",
    "ZhiYiDepthAnythingV2PreprocessorNodeV2",
]

DISPLAY_NAMES = [
    "知衣-RMBG2.0背景去除 v2",
    "知衣-衣物语义分割 v2",
    "知衣-时尚单品分割 v2",
    "知衣-身体部位分割 v2",
    "知衣-SAM2抠图 v2",
    "知衣-DWPose姿态检测 v2",
    "知衣-LineArt线稿预处理 v2",
    "知衣-DepthAnythingV2深度图预处理 v2",
]


def _image_data_url(size=(2, 2), color=(255, 0, 0)):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mask_data_url(size=(2, 2)):
    image = Image.new("L", size, 255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class DummyResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data
        self.text = text

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _segment_response(request_id="seg-1", size=(2, 2), classes=None):
    return {
        "image_size": {"width": size[0], "height": size[1]},
        "processed_image": _image_data_url(size=size, color=(0, 255, 0)),
        "mask": _mask_data_url(size=size),
        "mask_image": _image_data_url(size=size, color=(0, 0, 255)),
        "classes": classes or [],
        "request_id": request_id,
        "model_name": "rmbg",
        "device": "cpu",
        "inference_ms": 5.0,
        "total_ms": 10.0,
    }


def _pose_response(request_id="pose-1", size=(2, 2), people_count=1):
    return {
        "image_size": {"width": size[0], "height": size[1]},
        "pose_image": _image_data_url(size=size, color=(0, 255, 0)),
        "openpose_json": {"people": [{"pose_keypoints_2d": [1, 2, 0.9]}]},
        "people_count": people_count,
        "request_id": request_id,
        "model_name": "DWPose",
        "device": "cpu",
        "inference_ms": 12.3,
        "total_ms": 45.6,
    }


def _preprocess_response(request_id="pre-1", size=(2, 2)):
    return {
        "image_size": {"width": size[0], "height": size[1]},
        "processed_image": _image_data_url(size=size, color=(0, 255, 0)),
        "request_id": request_id,
        "preprocessor": "lineart",
        "model_name": "Lineart",
        "device": "cpu",
        "inference_ms": 5.0,
        "total_ms": 10.0,
    }


def _sam2_response(request_id="sam2-1", size=(2, 2)):
    return {
        "image_size": {"width": size[0], "height": size[1]},
        "mask": _mask_data_url(size=size),
        "count": 1,
        "masks": [{"score": 0.9}],
        "request_id": request_id,
    }


# ---------- 1. 注册与元数据 ----------


def test_v2_registration_and_metadata():
    for key, cls, name in zip(MAPPING_KEYS, V2_CLASSES, DISPLAY_NAMES):
        assert NODE_CLASS_MAPPINGS[key] is cls
        assert NODE_DISPLAY_NAME_MAPPINGS[key] == name

    for v2_cls, old_cls in zip(V2_CLASSES, OLD_CLASSES):
        assert v2_cls.RETURN_TYPES == old_cls.RETURN_TYPES
        assert v2_cls.RETURN_NAMES == old_cls.RETURN_NAMES
        assert v2_cls.FUNCTION == old_cls.FUNCTION
        assert v2_cls.CATEGORY == old_cls.CATEGORY
        assert v2_cls.OUTPUT_NODE == old_cls.OUTPUT_NODE


# ---------- 2. INPUT_TYPES 结构 ----------


def test_input_types_env_first_service_url_second():
    for cls in V2_CLASSES:
        input_types = cls.INPUT_TYPES()
        required_keys = list(input_types["required"])
        if cls is ZhiYiSAM2SegmentNodeV2:
            assert required_keys[0] == "images"
            assert required_keys[1] == "bboxes"
            assert required_keys[2] == "env"
            assert required_keys[3] == "service_url"
        else:
            assert required_keys[0] == "image"
            assert required_keys[1] == "env"
            assert required_keys[2] == "service_url"


def test_input_types_business_params_match_old():
    switch_filters = {
        ZhiYiDWPoseDetectNodeV2: {"return_pose_image", "return_openpose_json"},
        ZhiYiLineArtPreprocessorNodeV2: {"return_image"},
    }
    for v2_cls, old_cls in zip(V2_CLASSES, OLD_CLASSES):
        v2_keys = list(v2_cls.INPUT_TYPES()["required"])
        old_keys = list(old_cls.INPUT_TYPES()["required"])
        if v2_cls is ZhiYiSAM2SegmentNodeV2:
            assert v2_keys[4:] == old_keys[3:], v2_cls.__name__
        else:
            filtered = [key for key in v2_keys[3:] if key not in switch_filters.get(v2_cls, set())]
            assert filtered == old_keys[2:], v2_cls.__name__


def test_env_options_shape():
    values = [option["value"] for option in ENV_OPTIONS]
    assert values == ["gray", "dev", "custom"]
    for option in ENV_OPTIONS:
        assert set(option) == {"text", "value"}
        assert option["text"]
    assert CUSTOM_ENV == "custom"


def test_env_combo_is_object_options():
    for cls in V2_CLASSES:
        options, config = cls.INPUT_TYPES()["required"]["env"]
        assert all(isinstance(option, dict) and {"text", "value"} <= set(option) for option in options)


# ---------- 3. env 默认值 ----------


def test_env_defaults():
    rmbg_defaults = ["dev", "dev", "dev", "dev"]
    for cls, default in zip(
        [ZhiYiRMBGNodeV2, ZhiYiClothesSegmentNodeV2, ZhiYiFashionSegmentNodeV2, ZhiYiBodySegmentNodeV2],
        rmbg_defaults,
    ):
        assert cls.INPUT_TYPES()["required"]["env"][1]["default"] == default

    for cls in [ZhiYiDWPoseDetectNodeV2, ZhiYiLineArtPreprocessorNodeV2, ZhiYiDepthAnythingV2PreprocessorNodeV2, ZhiYiSAM2SegmentNodeV2]:
        assert cls.INPUT_TYPES()["required"]["env"][1]["default"] == "gray"


# ---------- 4. 多选 options ----------


def test_class_options_match_classes():
    for options, classes in [
        (CLOTHES_CLASS_OPTIONS, CLOTHES_CLASSES),
        (FASHION_CLASS_OPTIONS, FASHION_CLASSES),
        (BODY_CLASS_OPTIONS, BODY_CLASSES),
    ]:
        values = [option["value"] for option in options]
        assert set(values) == set(classes)
        assert len(values) == len(set(values)), "value 重复"
        assert all(option["text"] for option in options)


def test_class_defaults_are_valid():
    clothes = ZhiYiClothesSegmentNodeV2.INPUT_TYPES()["required"]["classes"]
    fashion = ZhiYiFashionSegmentNodeV2.INPUT_TYPES()["required"]["classes"]
    body = ZhiYiBodySegmentNodeV2.INPUT_TYPES()["required"]["classes"]

    assert clothes[1]["default"] == ["Upper-clothes"]
    assert fashion[1]["default"] == ["shirt, blouse"]
    assert body[1]["default"] == ["Face", "Hair", "Top-clothes", "Bottom-clothes"]

    assert all(item in CLOTHES_CLASSES for item in clothes[1]["default"])
    assert all(item in FASHION_CLASSES for item in fashion[1]["default"])
    assert all(item in BODY_CLASSES for item in body[1]["default"])


# ---------- 5. 开关默认值 ----------


def test_switch_defaults_true():
    dwpose_inputs = ZhiYiDWPoseDetectNodeV2.INPUT_TYPES()["required"]
    assert dwpose_inputs["return_pose_image"][1]["default"] is True
    assert dwpose_inputs["return_openpose_json"][1]["default"] is True

    lineart_inputs = ZhiYiLineArtPreprocessorNodeV2.INPUT_TYPES()["required"]
    assert lineart_inputs["return_image"][1]["default"] is True


# ---------- 6. resolve_service_env ----------


def test_resolve_service_env():
    assert resolve_service_env("gray", DWPOSE_SERVICE_ENVS) == DWPOSE_SERVICE_ENVS["gray"]
    assert resolve_service_env("dev", RMBG_SERVICE_ENVS) == RMBG_SERVICE_ENVS["dev"]
    assert resolve_service_env("gray", SAM2_SERVICE_ENVS) == SAM2_SERVICE_ENVS["gray"]
    assert resolve_service_env("dev", SAM2_SERVICE_ENVS) == SAM2_SERVICE_ENVS["dev"]
    assert resolve_service_env(None, DWPOSE_SERVICE_ENVS) is None
    assert resolve_service_env("custom", DWPOSE_SERVICE_ENVS) is None
    assert resolve_service_env("", DWPOSE_SERVICE_ENVS) is None

    with pytest.raises(RuntimeError, match="未知的 env: 不存在"):
        resolve_service_env("不存在", DWPOSE_SERVICE_ENVS)

    with pytest.raises(RuntimeError, match="未知的 env"):
        resolve_service_env({"text": "灰度环境", "value": "gray"}, DWPOSE_SERVICE_ENVS)

    with pytest.raises(RuntimeError, match="custom"):
        resolve_service_env("不存在", DWPOSE_SERVICE_ENVS)


# ---------- 7. 环境 URL 解析（mock 请求） ----------


def test_rmbg_v2_dev_env_url(monkeypatch):
    node = ZhiYiRMBGNodeV2()
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append(url)
        return DummyResponse(data=_segment_response())

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    node.remove_background(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        env="dev",
        service_url="https://custom.example.com/v1/rmbg",
        process_res=1024,
        sensitivity=1.0,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        refine_foreground=False,
        background="Alpha",
        background_color="#222222",
        return_image=True,
        return_mask=True,
        return_mask_image=False,
        max_concurrency=1,
        timeout=30,
    )
    assert calls == ["http://10.1.0.230:8003/v1/rmbg"]


def test_dwpose_v2_gray_env_url(monkeypatch):
    node = ZhiYiDWPoseDetectNodeV2()
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append(url)
        return DummyResponse(data=_pose_response())

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    node.detect_pose(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        env="gray",
        service_url="https://custom.example.com/v1/pose",
        resolution=512,
        detect_body=True,
        detect_hand=True,
        detect_face=False,
        xinsr_stick_scaling=False,
        return_pose_image=True,
        return_openpose_json=True,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
    )
    assert calls == ["http://model-api-dwpose-svc.online-server-gray:8001/v1/pose"]


def test_lineart_v2_custom_env_url(monkeypatch):
    node = ZhiYiLineArtPreprocessorNodeV2()
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append(url)
        return DummyResponse(data=_preprocess_response())

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    node.preprocess(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        env="custom",
        service_url="https://custom.example.com/v1/lineart",
        resolution=512,
        coarse=False,
        return_image=True,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
    )
    assert calls == ["https://custom.example.com/v1/lineart"]


def test_sam2_v2_dev_env_url(monkeypatch):
    node = ZhiYiSAM2SegmentNodeV2()
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append(url)
        return DummyResponse(data=_sam2_response())

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    node.segment(
        images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        bboxes=[[[0, 0, 2, 2]]],
        env="dev",
        service_url="https://custom.example.com/v1/segment",
        dilate=0,
        blur=0,
        background="Alpha",
        background_color="#222222",
        max_concurrency=1,
        timeout=30,
    )
    assert calls == ["http://10.1.0.230:8002/v1/segment"]


# ---------- 8. mock 请求 payload ----------


def test_clothes_v2_classes_array_in_payload(monkeypatch):
    node = ZhiYiClothesSegmentNodeV2()
    payloads = []

    def fake_request(method, url, timeout, **kwargs):
        payloads.append(kwargs["json"])
        return DummyResponse(data=_segment_response(classes=["Upper-clothes", "Pants"]))

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    node.segment(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        env="dev",
        service_url="",
        classes=["Upper-clothes", "Pants"],
        process_res=512,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        background="Alpha",
        background_color="#222222",
        return_image=True,
        return_mask=True,
        return_mask_image=False,
        max_concurrency=1,
        timeout=30,
    )
    assert payloads[0]["classes"] == ["Upper-clothes", "Pants"]


def test_clothes_v2_rejects_invalid_class(monkeypatch):
    node = ZhiYiClothesSegmentNodeV2()

    def fake_request(method, url, timeout, **kwargs):
        raise AssertionError("不应发出请求")

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="不支持的分割类别"):
        node.segment(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            env="dev",
            service_url="",
            classes=["Not-a-class"],
            process_res=512,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            background="Alpha",
            background_color="#222222",
            return_image=True,
            return_mask=True,
            return_mask_image=False,
            max_concurrency=1,
            timeout=30,
        )


def test_dwpose_v2_switches_false_payload_and_pass_through(monkeypatch):
    node = ZhiYiDWPoseDetectNodeV2()
    payloads = []

    def fake_request(method, url, timeout, **kwargs):
        payloads.append(kwargs["json"])
        return DummyResponse(data=_pose_response())

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    image = torch.rand((1, 2, 2, 3), dtype=torch.float32)
    pose_image, pose_keypoint, info_text = node.detect_pose(
        image=image,
        env="gray",
        service_url="",
        resolution=512,
        detect_body=True,
        detect_hand=True,
        detect_face=False,
        xinsr_stick_scaling=False,
        return_pose_image=False,
        return_openpose_json=False,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
    )
    assert payloads[0]["return_pose_image"] is False
    assert payloads[0]["return_openpose_json"] is False
    assert pose_image.shape == image.shape
    assert pose_image.dtype == image.dtype
    assert torch.equal(pose_image, image)
    assert pose_keypoint == [[]]
    assert json.loads(info_text)["return_pose_image"] is False
    assert json.loads(info_text)["return_openpose_json"] is False


def test_lineart_v2_return_image_false_pass_through(monkeypatch):
    node = ZhiYiLineArtPreprocessorNodeV2()
    payloads = []

    def fake_request(method, url, timeout, **kwargs):
        payloads.append(kwargs["json"])
        return DummyResponse(data={})

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    image = torch.rand((1, 2, 2, 3), dtype=torch.float32)
    result_image, info_text = node.preprocess(
        image=image,
        env="gray",
        service_url="",
        resolution=512,
        coarse=False,
        return_image=False,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
    )
    assert payloads[0]["return_image"] is False
    assert result_image.shape == image.shape
    assert result_image.dtype == image.dtype
    assert torch.equal(result_image, image)
    assert json.loads(info_text)["return_image"] is False


def test_sam2_v2_three_envs_post_urls(monkeypatch):
    node = ZhiYiSAM2SegmentNodeV2()
    urls = []

    def fake_request(method, url, timeout, **kwargs):
        urls.append(url)
        return DummyResponse(data=_sam2_response())

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    for env in ["gray", "dev", "custom"]:
        node.segment(
            images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            bboxes=[[[0, 0, 2, 2]]],
            env=env,
            service_url="https://custom.example.com/v1/segment",
            dilate=0,
            blur=0,
            background="Alpha",
            background_color="#222222",
            max_concurrency=1,
            timeout=30,
        )

    assert urls == [
        "http://model-api-sam2-hiera-base-plus-svc.online-server-gray:8000/v1/segment",
        "http://10.1.0.230:8002/v1/segment",
        "https://custom.example.com/v1/segment",
    ]


def test_rmbg_v2_invalid_env_raises(monkeypatch):
    node = ZhiYiRMBGNodeV2()

    def fake_request(method, url, timeout, **kwargs):
        raise AssertionError("不应发出请求")

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="未知的 env: 不存在"):
        node.remove_background(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            env="不存在",
            service_url="",
            process_res=1024,
            sensitivity=1.0,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Alpha",
            background_color="#222222",
            return_image=True,
            return_mask=True,
            return_mask_image=False,
            max_concurrency=1,
            timeout=30,
        )


# ---------- 9. 老节点回归保护 ----------


def test_old_dwpose_payload_still_returns_true():
    node = ZhiYiDWPoseDetectNode()
    payload = node._build_payload("data:image/png;base64,x", 512, True, True, False, False, "INTER_CUBIC")
    assert payload["return_pose_image"] is True
    assert payload["return_openpose_json"] is True


def test_old_controlnet_payload_still_returns_image_true():
    node = ZhiYiLineArtPreprocessorNode()
    payload = node._build_common_payload("data:image/png;base64,x", 512, "INTER_CUBIC")
    assert payload["return_image"] is True


def test_old_nodes_unchanged_input_types():
    for old_cls in OLD_CLASSES:
        input_types = old_cls.INPUT_TYPES()
        assert "env" not in input_types["required"]
        assert list(input_types["required"])[1] == "bboxes" or list(input_types["required"])[1] == "service_url"


def test_v2_node_switch_short_circuit(monkeypatch):
    def assert_no_request(method, url, timeout, **kwargs):
        raise AssertionError("不应发出请求")

    rmbg_node = ZhiYiRMBGNodeV2()
    monkeypatch.setattr(rmbg_module.requests, "request", assert_no_request)
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    result_image, mask, mask_image, info_text = rmbg_node.remove_background(
        image=image,
        env="dev",
        service_url="",
        process_res=1024,
        sensitivity=1.0,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        refine_foreground=False,
        background="Alpha",
        background_color="#222222",
        return_image=True,
        return_mask=True,
        return_mask_image=False,
        max_concurrency=1,
        timeout=30,
        node_switch=1,
    )
    assert torch.equal(result_image, image)
    assert json.loads(info_text)["skipped"] is True

    dwpose_node = ZhiYiDWPoseDetectNodeV2()
    pose_image, keypoints, info_text = dwpose_node.detect_pose(
        image=image,
        env="不存在",
        service_url="",
        resolution=512,
        detect_body=True,
        detect_hand=True,
        detect_face=False,
        xinsr_stick_scaling=False,
        return_pose_image=True,
        return_openpose_json=True,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        node_switch=1,
    )
    assert torch.equal(pose_image, image)
    assert keypoints == []
    assert json.loads(info_text)["skipped"] is True


def test_v2_background_color_validation(monkeypatch):
    node = ZhiYiRMBGNodeV2()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data=_segment_response())

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="background_color 不是有效 hex 颜色"):
        node.remove_background(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            env="dev",
            service_url="",
            process_res=1024,
            sensitivity=1.0,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Color",
            background_color="#zzzzzz",
            return_image=True,
            return_mask=True,
            return_mask_image=False,
            max_concurrency=1,
            timeout=30,
        )


def test_sam2_v2_node_switch(monkeypatch):
    def assert_no_request(method, url, timeout, **kwargs):
        raise AssertionError("不应发出请求")

    monkeypatch.setattr(sam2_module.requests, "request", assert_no_request)
    node = ZhiYiSAM2SegmentNodeV2()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    result_image, mask, mask_image, info_text = node.segment(
        images=image,
        bboxes=[[[0, 0, 2, 2]]],
        env="不存在",
        service_url="",
        dilate=0,
        blur=0,
        background="Alpha",
        background_color="#222222",
        max_concurrency=1,
        timeout=30,
        node_switch=1,
    )
    assert torch.equal(result_image, image)
    assert mask.shape == (1, 2, 2)
    assert mask_image.shape == (1, 2, 2, 3)
    assert json.loads(info_text)["skipped"] is True


def test_sam2_v2_empty_bbox_pass_through_no_request(monkeypatch):
    def assert_no_request(method, url, timeout, **kwargs):
        raise AssertionError("不应发出请求")

    monkeypatch.setattr(sam2_module.requests, "request", assert_no_request)
    node = ZhiYiSAM2SegmentNodeV2()
    image = torch.ones((1, 2, 2, 3), dtype=torch.float32)
    result_image, mask, mask_image, info_text = node.segment(
        images=image,
        bboxes=[],
        env="gray",
        service_url="",
        dilate=0,
        blur=0,
        background="Alpha",
        background_color="#222222",
        max_concurrency=1,
        timeout=30,
    )
    assert torch.equal(result_image, image)
    assert mask.shape == (1, 2, 2)
    assert json.loads(info_text)["items"][0]["skipped"] is True


def test_custom_env_falls_back_to_fd_var(monkeypatch):
    """env=custom 时保留老节点的 FD_* 环境变量兜底行为。"""
    node = ZhiYiDWPoseDetectNodeV2()
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append(url)
        return DummyResponse(data=_pose_response())

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)
    monkeypatch.setattr(dwpose_module, "FD_DWPOSE_POSE_URL", "http://fd-env.example.com:9000/v1/pose")

    node.detect_pose(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        env="custom",
        service_url="",
        resolution=512,
        detect_body=True,
        detect_hand=True,
        detect_face=False,
        xinsr_stick_scaling=False,
        return_pose_image=True,
        return_openpose_json=True,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
    )
    assert calls == ["http://fd-env.example.com:9000/v1/pose"]


# ---------- Review 修复后的补充测试 ----------


def test_class_widgets_have_multiselect_markers():
    for cls, _ in [
        (ZhiYiClothesSegmentNodeV2, CLOTHES_CLASS_OPTIONS),
        (ZhiYiFashionSegmentNodeV2, FASHION_CLASS_OPTIONS),
        (ZhiYiBodySegmentNodeV2, BODY_CLASS_OPTIONS),
    ]:
        options, config = cls.INPUT_TYPES()["required"]["classes"]
        assert config["multiselect"] is True
        assert config["multi_select"]
        assert all(isinstance(o, dict) and {"text", "value"} <= set(o) for o in options)


def test_sam2_v2_node_switch_batch_mask(monkeypatch):
    def assert_no_request(method, url, timeout, **kwargs):
        raise AssertionError("不应发出请求")

    monkeypatch.setattr(sam2_module.requests, "request", assert_no_request)
    node = ZhiYiSAM2SegmentNodeV2()
    images = torch.rand((3, 2, 2, 3), dtype=torch.float32)
    result_image, mask, mask_image, info_text = node.segment(
        images=images,
        bboxes=[[[0, 0, 2, 2]]],
        env="不存在",
        service_url="",
        dilate=0,
        blur=0,
        background="Alpha",
        background_color="#222222",
        max_concurrency=1,
        timeout=30,
        node_switch=1,
    )
    assert result_image.shape == (3, 2, 2, 3)
    assert mask.shape == (3, 2, 2)
    assert mask.dtype == images.dtype
    assert mask.device == images.device
    assert mask_image.shape == (3, 2, 2, 3)
    assert torch.equal(result_image, images)
    assert json.loads(info_text)["skipped"] is True


def test_dwpose_v2_service_url_default_reads_env(monkeypatch):
    monkeypatch.setattr(aux_v2_module, "FD_DWPOSE_POSE_URL", "http://fd-env.example.com:9000/v1/pose")
    default = ZhiYiDWPoseDetectNodeV2.INPUT_TYPES()["required"]["service_url"][1]["default"]
    assert default == "http://fd-env.example.com:9000/v1/pose"


def test_dwpose_v2_gray_env_overrides_env_default(monkeypatch):
    monkeypatch.setattr(aux_v2_module, "FD_DWPOSE_POSE_URL", "http://fd-env.example.com:9000/v1/pose")
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append(url)
        return DummyResponse(data=_pose_response())

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)
    node = ZhiYiDWPoseDetectNodeV2()
    node.detect_pose(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        env="gray",
        service_url="http://fd-env.example.com:9000/v1/pose",
        resolution=512,
        detect_body=True,
        detect_hand=True,
        detect_face=False,
        xinsr_stick_scaling=False,
        return_pose_image=True,
        return_openpose_json=True,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
    )
    assert calls == ["http://model-api-dwpose-svc.online-server-gray:8001/v1/pose"]


def test_sam2_v2_invalid_background_zero_requests(monkeypatch):
    def assert_no_request(method, url, timeout, **kwargs):
        raise AssertionError("不应发出请求")

    monkeypatch.setattr(sam2_module.requests, "request", assert_no_request)
    node = ZhiYiSAM2SegmentNodeV2()
    with pytest.raises(RuntimeError, match="background 无效"):
        node.segment(
            images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            bboxes=[[[0, 0, 2, 2]]],
            env="gray",
            service_url="",
            dilate=0,
            blur=0,
            background="NotARealBackground",
            background_color="#222222",
            max_concurrency=1,
            timeout=30,
        )


def test_sam2_v2_invalid_color_zero_requests(monkeypatch):
    def assert_no_request(method, url, timeout, **kwargs):
        raise AssertionError("不应发出请求")

    monkeypatch.setattr(sam2_module.requests, "request", assert_no_request)
    node = ZhiYiSAM2SegmentNodeV2()
    with pytest.raises(RuntimeError, match="background_color 不是有效 hex 颜色"):
        node.segment(
            images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            bboxes=[[[0, 0, 2, 2]]],
            env="gray",
            service_url="",
            dilate=0,
            blur=0,
            background="Color",
            background_color="#zzzzzz",
            max_concurrency=1,
            timeout=30,
        )


def test_class_options_rejects_bad_tables():
    with pytest.raises(RuntimeError, match="缺少翻译"):
        _build_class_options(["a", "b"], {"a": "甲"})
    with pytest.raises(RuntimeError, match="多余翻译"):
        _build_class_options(["a"], {"a": "甲", "b": "乙"})
    with pytest.raises(RuntimeError, match="类别重复"):
        _build_class_options(["a", "a"], {"a": "甲"})
    with pytest.raises(RuntimeError, match="空翻译"):
        _build_class_options(["a"], {"a": ""})
