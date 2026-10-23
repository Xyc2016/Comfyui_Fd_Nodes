"""FD_imgToText_Doubao 兼容 image-generation image_to_text_doubao 响应格式的测试。"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# old_fd_nodes 依赖 utils.common_util / utils.oss_client（间接依赖 torch/PIL），
# 本测试只覆盖 HTTP 协议兼容，直接 stub 掉这两个工具模块
for mod_name, attrs in (
    ("Comfyui_Fd_Nodes.utils.common_util", ("bytes_calculate_hex_md5", "pil2iobyte", "tensor2pil")),
    ("Comfyui_Fd_Nodes.utils.oss_client", ("upload_bytes_to_oss",)),
):
    if mod_name not in sys.modules:
        fake = types.ModuleType(mod_name)
        for attr in attrs:
            setattr(fake, attr, lambda *a, **k: None)
        sys.modules[mod_name] = fake

from Comfyui_Fd_Nodes import old_fd_nodes  # noqa: E402


class _Resp:
    def __init__(self, payload):
        self.content = json.dumps(payload).encode()


def _gen(payload):
    with patch.object(old_fd_nodes.requests, "post", return_value=_Resp(payload)) as post:
        out = old_fd_nodes.FD_imgToText_Doubao().gen(
            "https://example.com/a.png", "描述图片", "fallback")[0]
        sent = json.loads(post.call_args.kwargs.get("data") or post.call_args.args[-1]
                          if post.call_args.kwargs.get("data") else post.call_args[1].get("data", "{}"))
        return out, sent, post.call_args.kwargs.get("headers", {})


def test_legacy_gateway_response():
    out, sent, headers = _gen({"status": True, "response": {"Result": "A woman in her 20s."}})
    assert out == "A woman in her 20s."
    assert sent["image_url"] == "https://example.com/a.png"
    assert sent["text_prompt"] == "描述图片"


def test_image_generation_doubao_response():
    out, _, _ = _gen({
        "status": True, "model": "doubao-seed-2-0-mini",
        "result_text": "A woman in her 20s.", "cost_time": 2.3,
    })
    assert out == "A woman in her 20s."


def test_no_auth_header_when_key_missing():
    with patch.object(old_fd_nodes, "FD_DOUBAO_KEY", None):
        _, _, headers = _gen({"status": True, "result_text": "ok"})
    assert "Authorization" not in headers


def test_auth_header_when_key_present():
    with patch.object(old_fd_nodes, "FD_DOUBAO_KEY", "secret-key"):
        _, _, headers = _gen({"status": True, "response": {"Result": "ok"}})
    assert headers.get("Authorization") == "Bearer secret-key"


def test_status_false_falls_back_to_default():
    out, _, _ = _gen({"status": False, "error": {"code": "TIMEOUT", "message": "x"}})
    assert out == "fallback"


def test_malformed_response_falls_back_to_default():
    out, _, _ = _gen({"unrelated": True})
    assert out == "fallback"
