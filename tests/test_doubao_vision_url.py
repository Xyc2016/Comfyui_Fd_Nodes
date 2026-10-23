"""FD_DOUBAO_VISION_URL 配置时图生文节点走 image-generation 的行为测试。"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 图生文节点间接依赖 torch（numpy tensor 输入），测试只覆盖请求构造与响应解析，
# stub 掉 torch 后用手造的伪 tensor（带 detach().cpu().numpy() 链）驱动压缩分支
if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    sys.modules["torch"] = fake_torch

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from Comfyui_Fd_Nodes import (  # noqa: E402
    zhiyi_image_text_combo_node as combo_mod,
    zhiyi_image_text_node as single_mod,
)


class _FakeTensor:
    """最小 tensor 桩：满足 _image_tensor_to_pil 与 _expand_images 的处理链。"""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.ndim = arr.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, idx):
        return _FakeTensor(self._arr[idx])


def _tensor(color, shape=(1, 64, 64, 3)):
    arr = np.zeros(shape, dtype=np.float32)
    arr[..., 0] = color[0] / 255.0
    arr[..., 1] = color[1] / 255.0
    arr[..., 2] = color[2] / 255.0
    return _FakeTensor(arr)


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


VISION_URL = "http://image-generation.gray.svc.cluster.local:8207/image/image_to_text_doubao"


def test_single_image_via_vision_url():
    node = single_mod.ZhiYiImageTextNode()
    resp = _Resp({"status": True, "result_text": "红色方块", "model": "doubao-seed-2-0-mini"})
    with patch.object(single_mod, "FD_DOUBAO_VISION_URL", VISION_URL), \
            patch.object(single_mod.requests, "post", return_value=resp) as post:
        out = node.generate(_tensor((200, 60, 60)), "什么颜色", node_switch=0)
    assert out[0] == "红色方块"
    sent = post.call_args.kwargs["json"]
    assert sent["text_prompt"] == "什么颜色"
    assert "image_base64" in sent and sent["image_base64"]
    assert post.call_args.kwargs["url"] == VISION_URL


def test_single_image_system_prompt_merged():
    node = single_mod.ZhiYiImageTextNode()
    resp = _Resp({"status": True, "result_text": "ok"})
    with patch.object(single_mod, "FD_DOUBAO_VISION_URL", VISION_URL), \
            patch.object(single_mod.requests, "post", return_value=resp) as post:
        node.generate(_tensor((60, 60, 200)), "描述", node_switch=0, system_prompt="你是商品描述助手")
    assert post.call_args.kwargs["json"]["text_prompt"].startswith("你是商品描述助手\n")


def test_single_image_service_error_raises():
    node = single_mod.ZhiYiImageTextNode()
    resp = _Resp({"status": False, "error": {"code": "TIMEOUT", "message": "upstream timeout"}})
    with patch.object(single_mod, "FD_DOUBAO_VISION_URL", VISION_URL), \
            patch.object(single_mod.requests, "post", return_value=resp):
        with pytest.raises(RuntimeError, match="TIMEOUT"):
            node.generate(_tensor((10, 10, 10)), "描述", node_switch=0)


def test_single_image_fallback_to_litellm_when_unset():
    node = single_mod.ZhiYiImageTextNode()
    litellm_payload = {
        "choices": [{"message": {"content": "litellm 文本"}}],
    }

    class _LiteLLMResp(_Resp):
        def __init__(self):
            super().__init__(litellm_payload)

    cfg = {"base_url": "http://litellm:4000", "api_key": "sk-test"}
    with patch.object(single_mod, "FD_DOUBAO_VISION_URL", ""), \
            patch.object(single_mod, "load_config", return_value=cfg), \
            patch.object(single_mod.requests, "post", return_value=_LiteLLMResp()) as post:
        out = node.generate(_tensor((90, 90, 90)), "介绍", node_switch=0)
    assert out[0] == "litellm 文本"
    assert post.call_args.kwargs["url"] == "http://litellm:4000/v1/chat/completions"


def test_combo_multi_image_via_vision_url():
    node = combo_mod.ZhiYiImageTextComboNode()
    resp = _Resp({"status": True, "result_text": "第一张红色，第二张蓝色", "model": "doubao-seed-2-0-mini"})
    combo = {"images": [_tensor((200, 60, 60)), _tensor((60, 60, 200))], "prompt": "分别什么颜色"}
    with patch.object(combo_mod, "FD_DOUBAO_VISION_URL", VISION_URL), \
            patch.object(combo_mod.requests, "post", return_value=resp) as post:
        text, log = node.generate(
            node_switch=0, combo_1=combo, max_concurrency=2, retry_count=1,
        )
    assert "红色" in text
    sent = post.call_args.kwargs["json"]
    assert len(sent["image_base64_list"]) == 2
    assert sent["text_prompt"] == "分别什么颜色"
    assert post.call_args.kwargs["url"] == VISION_URL


def test_combo_service_error_marks_slot_failed():
    node = combo_mod.ZhiYiImageTextComboNode()
    resp = _Resp({"status": False, "error": {"code": "INVALID_PARAMS", "message": "bad image"}})
    combo = {"images": [_tensor((1, 2, 3))], "prompt": "描述"}
    with patch.object(combo_mod, "FD_DOUBAO_VISION_URL", VISION_URL), \
            patch.object(combo_mod.requests, "post", return_value=resp):
        with pytest.raises(RuntimeError):
            node.generate(node_switch=0, combo_1=combo, retry_count=1)
