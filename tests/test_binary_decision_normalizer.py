#!/usr/bin/env python3
"""FD_BinaryDecisionNormalizer 与 normalize_binary_decision 离线测试。

任何输入下输出都必须是 int 且值属于 {0, 1}，保证下游
easy anythingIndexSwitch 的 index 永远不会是 None 或浮点。
"""

import os
import unittest
from unittest.mock import Mock, patch

# 环境变量须在 config 导入前就位；强制赋值以免其他测试先导入
os.environ["FD_DOUBAO_KEY"] = os.environ.get("FD_DOUBAO_KEY") or "test-key"
os.environ["FD_DOUBAO_URL"] = os.environ.get("FD_DOUBAO_URL") or "https://example.com/v1/chat"

from src.Comfyui_Fd_Nodes.old_fd_nodes import (  # noqa: E402
    FD_BinaryDecisionNormalizer,
    FD_imgToText_Doubao,
    normalize_binary_decision,
)


class NormalizeBinaryDecisionTest(unittest.TestCase):
    def assert_binary(self, value, expected=None):
        result = normalize_binary_decision(value)
        self.assertIs(type(result), int, f"value={value!r} result={result!r}")
        self.assertIn(result, (0, 1), f"value={value!r} result={result!r}")
        if expected is not None:
            self.assertEqual(result, expected, f"value={value!r}")

    def test_plain_binary(self):
        self.assert_binary("0", 0)
        self.assert_binary("1", 1)

    def test_int_and_float_inputs(self):
        self.assert_binary(0, 0)
        self.assert_binary(1, 1)
        self.assert_binary(0.0, 0)
        self.assert_binary(1.0, 1)
        self.assert_binary(True, 1)
        self.assert_binary(False, 0)

    def test_whitespace_and_newlines(self):
        self.assert_binary("  0  ", 0)
        self.assert_binary("\n1\n", 1)
        self.assert_binary("\t0\t", 0)

    def test_markdown_code_fence(self):
        self.assert_binary("```\n0\n```", 0)
        self.assert_binary("```text\n1\n```", 1)
        self.assert_binary("```\n0", 0)

    def test_single_token_explanation(self):
        self.assert_binary("无 logo 叠加，结果：0", 0)
        self.assert_binary("结论是 1（有水印）", 1)
        self.assert_binary("图中无任何后期叠加标识，返回 0。", 0)

    def test_empty_and_none(self):
        for value in (None, "", "   ", "\n\t\n", "```\n```", "```\n   \n```"):
            result = normalize_binary_decision(value)
            self.assertIs(type(result), int)
            self.assertIn(result, (0, 1))

    def test_default_value_used_on_empty(self):
        self.assertEqual(normalize_binary_decision(None, 1), 1)
        self.assertEqual(normalize_binary_decision("", 1), 1)
        self.assertEqual(normalize_binary_decision("   ", 0), 0)

    def test_ambiguous_text_falls_back(self):
        for value in ("0 与 1 均存在", "10", "01", "1.0", "0.0", "101"):
            result = normalize_binary_decision(value)
            self.assertIs(type(result), int)
            self.assertIn(result, (0, 1))
            self.assertEqual(result, 0)

    def test_ambiguous_text_uses_given_default(self):
        self.assertEqual(normalize_binary_decision("0 与 1 均存在", 1), 1)
        self.assertEqual(normalize_binary_decision("10", 1), 1)
        self.assertEqual(normalize_binary_decision("1.0", 1), 1)
        self.assertEqual(normalize_binary_decision("2", 1), 1)

    def test_non_binary_value_falls_back(self):
        for value in ("2", "-1", "0.5", "abc", "yes", "watermark", ["0"]):
            result = normalize_binary_decision(value)
            self.assertIs(type(result), int)
            self.assertIn(result, (0, 1))

    def test_default_value_out_of_range_coerced(self):
        self.assertEqual(normalize_binary_decision("", 7), 0)
        self.assertEqual(normalize_binary_decision(None, -1), 0)


class BinaryDecisionNormalizerNodeTest(unittest.TestCase):
    def setUp(self):
        self.node = FD_BinaryDecisionNormalizer()

    def test_return_type_declared_int(self):
        self.assertEqual(FD_BinaryDecisionNormalizer.RETURN_TYPES, ("INT",))

    def test_plain_inputs(self):
        self.assertEqual(self.node.normalize("0"), (0,))
        self.assertEqual(self.node.normalize("1"), (1,))

    def test_explanation_and_fence(self):
        self.assertEqual(self.node.normalize("检测无任何水印，结论 0"), (0,))
        self.assertEqual(self.node.normalize("```\n1\n```"), (1,))
        self.assertEqual(self.node.normalize("\n0\n"), (0,))

    def test_garbage_uses_default_value(self):
        self.assertEqual(self.node.normalize("", 0), (0,))
        self.assertEqual(self.node.normalize("10", 0), (0,))
        self.assertEqual(self.node.normalize("1.0", 0), (0,))
        self.assertEqual(self.node.normalize(None, 1), (1,))
        self.assertEqual(self.node.normalize("10", 1), (1,))
        self.assertEqual(self.node.normalize("2", 1), (1,))

    def test_node_output_always_binary_int(self):
        for value in (None, "", "0", "1", "10", "1.0", "```\n0\n```", "结论：0", "abc"):
            result = self.node.normalize(value, 0)[0]
            self.assertIs(type(result), int)
            self.assertIn(result, (0, 1))


class DoubaoNodeErrorBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.node = FD_imgToText_Doubao()

    def _patch_post(self, response):
        return patch(
            "src.Comfyui_Fd_Nodes.old_fd_nodes.requests.post",
            return_value=response,
        )

    def test_transport_exception_returns_default_prompt(self):
        with patch("src.Comfyui_Fd_Nodes.old_fd_nodes.requests.post", side_effect=RuntimeError("connection refused")):
            self.assertEqual(self.node.gen("http://img", "p", "fallback"), ("fallback",))

    def test_non_json_response_returns_default_prompt(self):
        response = Mock()
        response.content = b"<html>gateway error</html>"
        with self._patch_post(response):
            self.assertEqual(self.node.gen("http://img", "p", "fallback"), ("fallback",))

    def test_failed_status_returns_default_prompt(self):
        response = Mock()
        response.content = b'{"status": false, "response": {"Result": "oops"}}'
        with self._patch_post(response):
            self.assertEqual(self.node.gen("http://img", "p", "fallback"), ("fallback",))

    def test_missing_field_returns_default_prompt(self):
        response = Mock()
        response.content = b'{"status": true}'
        with self._patch_post(response):
            self.assertEqual(self.node.gen("http://img", "p", "fallback"), ("fallback",))

    def test_success_returns_vlm_result(self):
        response = Mock()
        response.content = b'{"status": true, "response": {"Result": "1"}}'
        with self._patch_post(response):
            self.assertEqual(self.node.gen("http://img", "p", "fallback"), ("1",))


if __name__ == "__main__":
    unittest.main()
