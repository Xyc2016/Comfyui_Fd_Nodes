"""Tests for zhiyi_qwen_detect_node parsing logic (no real API calls)."""

import json
from unittest.mock import Mock, call

import pytest
import requests
import torch

from src.Comfyui_Fd_Nodes import zhiyi_qwen_detect_node as qwen_module
from src.Comfyui_Fd_Nodes.zhiyi_qwen_detect_node import (
    ZhiYiBBoxesToSAM2,
    ZhiYiQwenDetectNode,
    _build_chat_url,
    _extract_json_text,
    _normalize_box,
    _safe_score,
    parse_boxes,
)


# ---------------------------------------------------------------------------
# _extract_json_text
# ---------------------------------------------------------------------------

class TestExtractJsonText:
    def test_plain_json_array(self):
        raw = '[{"bbox_2d":[100,200,500,800],"label":"cat"}]'
        assert _extract_json_text(raw) == raw

    def test_json_in_markdown_fence(self):
        raw = '```json\n[{"bbox_2d":[1,2,3,4]}]\n```'
        assert _extract_json_text(raw) == '[{"bbox_2d":[1,2,3,4]}]'

    def test_json_in_plain_fence(self):
        raw = '```\n[{"bbox": [1,2,3,4]}]\n```'
        assert _extract_json_text(raw) == '[{"bbox": [1,2,3,4]}]'

    def test_content_dict_with_list(self):
        raw = '{"content": [{"bbox_2d": [10,20,30,40]}]}'
        assert _extract_json_text(raw) == '[{"bbox_2d": [10, 20, 30, 40]}]'

    def test_content_dict_with_string(self):
        raw = '{"content": "[{\\"bbox_2d\\": [10,20,30,40]}]"}'
        assert _extract_json_text(raw) == '[{"bbox_2d": [10,20,30,40]}]'

    def test_explanation_before_json(self):
        raw = 'I found 2 objects.\n[{"bbox_2d":[1,2,3,4]},{"bbox_2d":[5,6,7,8]}]'
        result = _extract_json_text(raw)
        assert result.startswith("[{")

    def test_explanation_after_json(self):
        raw = '[{"bbox_2d":[1,2,3,4]}]\nThese are the detected objects.'
        result = _extract_json_text(raw)
        parsed = json.loads(result)
        assert len(parsed) == 1

    def test_bracket_inside_string_not_truncated(self):
        raw = '[{"bbox_2d":[1,2,3,4],"label":"a]b"}] trailing text'
        result = _extract_json_text(raw)
        parsed = json.loads(result)
        assert parsed[0]["label"] == "a]b"

    def test_brace_inside_string_not_truncated(self):
        raw = '[{"bbox_2d":[1,2,3,4],"label":"a}b"}] trailing'
        result = _extract_json_text(raw)
        parsed = json.loads(result)
        assert parsed[0]["label"] == "a}b"

    def test_escaped_quote_inside_string(self):
        raw = '[{"bbox_2d":[1,2,3,4],"label":"a\\"b"}] extra'
        result = _extract_json_text(raw)
        parsed = json.loads(result)
        assert parsed[0]["label"] == 'a"b'


# ---------------------------------------------------------------------------
# _normalize_box — coordinate scale detection
# ---------------------------------------------------------------------------

class TestNormalizeBox:
    W, H = 800, 600

    def test_0_1_normalized_auto(self):
        box = _normalize_box([0.1, 0.2, 0.5, 0.8], self.W, self.H)
        assert box == [80, 120, 400, 480]

    def test_0_1_normalized_explicit(self):
        box = _normalize_box([0.1, 0.2, 0.5, 0.8], self.W, self.H, coord_format="01")
        assert box == [80, 120, 400, 480]

    def test_0_1000_normalized_auto(self):
        box = _normalize_box([100, 200, 500, 800], self.W, self.H)
        assert box == [80, 120, 400, 480]

    def test_0_1000_normalized_explicit(self):
        box = _normalize_box([0.1, 0.2, 0.5, 0.8], self.W, self.H, coord_format="1000")
        assert box == [0, 0, 0, 0]

    def test_pixel_coordinates_auto(self):
        box = _normalize_box([1200, 1800, 3000, 4500], self.W, self.H)
        assert box == [800, 600, 800, 600]

    def test_pixel_coordinates_explicit(self):
        box = _normalize_box([100, 200, 500, 800], self.W, self.H, coord_format="pixel")
        assert box == [100, 200, 500, 600]

    def test_pixel_coordinates_small_image_floats(self):
        box = _normalize_box([5.5, 10.2, 15.8, 20.1], 32, 32)
        assert box == [6, 10, 16, 20]

    def test_pixel_coordinates_small_image_explicit(self):
        # Integer pixel coords on 32x32 — auto would misclassify as 0-1000
        box = _normalize_box([5, 10, 15, 20], 32, 32, coord_format="pixel")
        assert box == [5, 10, 15, 20]

    def test_reversed_box_fixed(self):
        box = _normalize_box([0.5, 0.8, 0.1, 0.2], self.W, self.H)
        assert box == [80, 120, 400, 480]

    def test_clamp_to_image_bounds(self):
        box = _normalize_box([0.0, 0.0, 1.0, 1.0], self.W, self.H)
        assert box == [0, 0, 800, 600]

    def test_negative_clamped(self):
        box = _normalize_box([-0.1, -0.1, 0.5, 0.5], self.W, self.H)
        assert box == [0, 0, 400, 300]


# ---------------------------------------------------------------------------
# _safe_score
# ---------------------------------------------------------------------------

class TestSafeScore:
    def test_valid_float(self):
        assert _safe_score(0.85) == 0.85

    def test_string_number(self):
        assert _safe_score("0.9") == 0.9

    def test_none_default(self):
        assert _safe_score(None) == 1.0

    def test_nan_default(self):
        assert _safe_score(float("nan")) == 1.0

    def test_non_numeric_string(self):
        assert _safe_score("high") == 1.0

    def test_null_json(self):
        assert _safe_score(None) == 1.0


# ---------------------------------------------------------------------------
# parse_boxes — full integration
# ---------------------------------------------------------------------------

class TestParseBoxes:
    W, H = 800, 600

    def test_direct_json_array_with_bbox_2d(self):
        text = '[{"bbox_2d":[100,200,500,800],"label":"person","score":0.9}]'
        result = parse_boxes(text, self.W, self.H)
        assert len(result) == 1
        assert result[0]["bbox"] == [80, 120, 400, 480]
        assert result[0]["score"] == 0.9
        assert result[0]["label"] == "person"

    def test_bbox_field_name_variants(self):
        for field in ("bbox_2d", "bbox", "box_2d", "box"):
            text = json.dumps([{field: [100, 200, 500, 800], "label": "x"}])
            result = parse_boxes(text, self.W, self.H)
            assert len(result) == 1, f"Failed for field {field}"

    def test_item_is_plain_list(self):
        text = '[[100, 200, 500, 800]]'
        result = parse_boxes(text, self.W, self.H)
        assert len(result) == 1
        assert result[0]["bbox"] == [80, 120, 400, 480]

    def test_content_wrapper_dict(self):
        text = '{"content": [{"bbox_2d": [100, 200, 500, 800]}]}'
        result = parse_boxes(text, self.W, self.H)
        assert len(result) == 1

    def test_content_string_wrapper(self):
        text = '{"content": "[{\\"bbox_2d\\": [100, 200, 500, 800]}]"}'
        result = parse_boxes(text, self.W, self.H)
        assert len(result) == 1

    def test_score_threshold_filters(self):
        text = '[{"bbox_2d":[1,2,3,4],"score":0.3},{"bbox_2d":[5,6,7,8],"score":0.8}]'
        result = parse_boxes(text, self.W, self.H, score_threshold=0.5)
        assert len(result) == 1
        assert result[0]["score"] == 0.8

    def test_default_score_is_1(self):
        text = '[{"bbox_2d":[1,2,3,4]}]'
        result = parse_boxes(text, self.W, self.H, score_threshold=0.5)
        assert len(result) == 1

    def test_sorted_by_score_descending(self):
        text = '[{"bbox_2d":[1,2,3,4],"score":0.3},{"bbox_2d":[5,6,7,8],"score":0.9}]'
        result = parse_boxes(text, self.W, self.H)
        assert result[0]["score"] == 0.9
        assert result[1]["score"] == 0.3

    def test_empty_result_on_unparseable(self):
        result = parse_boxes("I cannot find any objects.", self.W, self.H)
        assert result == []

    def test_truncated_json(self):
        text = '[{"bbox_2d":[100,200,500,800],"label":"a"},{"bbox_2d":[10,20,30,40'
        result = parse_boxes(text, self.W, self.H)
        assert len(result) >= 1

    def test_multiple_objects(self):
        items = [
            {"bbox_2d": [100, 100, 300, 300], "label": "cat", "score": 0.9},
            {"bbox_2d": [500, 200, 700, 500], "label": "dog", "score": 0.7},
        ]
        text = json.dumps(items)
        result = parse_boxes(text, self.W, self.H)
        assert len(result) == 2

    def test_markdown_wrapped(self):
        inner = json.dumps([{"bbox_2d": [100, 200, 500, 800], "label": "car"}])
        text = f"```json\n{inner}\n```"
        result = parse_boxes(text, self.W, self.H)
        assert len(result) == 1

    def test_malformed_score_defaults_to_1(self):
        text = '[{"bbox_2d":[1,2,3,4],"score":null},{"bbox_2d":[5,6,7,8],"score":"high"}]'
        result = parse_boxes(text, self.W, self.H)
        assert len(result) == 2
        assert result[0]["score"] == 1.0
        assert result[1]["score"] == 1.0

    def test_coord_format_pixel_overrides_auto(self):
        text = '[[5, 10, 15, 20]]'
        result = parse_boxes(text, 32, 32, coord_format="pixel")
        assert len(result) == 1
        assert result[0]["bbox"] == [5, 10, 15, 20]

    def test_coord_format_1000_explicit(self):
        text = '[[100, 200, 500, 800]]'
        result = parse_boxes(text, self.W, self.H, coord_format="1000")
        assert len(result) == 1
        assert result[0]["bbox"] == [80, 120, 400, 480]


# ---------------------------------------------------------------------------
# _build_chat_url
# ---------------------------------------------------------------------------

class TestBuildChatUrl:
    def test_base_without_v1(self):
        assert _build_chat_url("https://api.example.com") == "https://api.example.com/v1/chat/completions"

    def test_base_with_v1(self):
        assert _build_chat_url("https://api.example.com/v1") == "https://api.example.com/v1/chat/completions"

    def test_trailing_slash_without_v1(self):
        assert _build_chat_url("https://api.example.com/") == "https://api.example.com/v1/chat/completions"

    def test_trailing_slash_with_v1(self):
        assert _build_chat_url("https://api.example.com/v1/") == "https://api.example.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# ZhiYiBBoxesToSAM2
# ---------------------------------------------------------------------------

class TestBBoxesToSAM2:
    def test_wraps_flat_list(self):
        node = ZhiYiBBoxesToSAM2()
        result = node.convert([[10, 20, 30, 40], [50, 60, 70, 80]])
        assert result == ([[[10, 20, 30, 40], [50, 60, 70, 80]]],)

    def test_already_batched(self):
        node = ZhiYiBBoxesToSAM2()
        bboxes = [[[10, 20, 30, 40], [50, 60, 70, 80]]]
        result = node.convert(bboxes)
        assert result == (bboxes,)

    def test_empty_list(self):
        node = ZhiYiBBoxesToSAM2()
        result = node.convert([])
        assert result == ([[]],)


# ---------------------------------------------------------------------------
# ZhiYiQwenDetectNode metadata
# ---------------------------------------------------------------------------

class TestQwenDetectNodeMetadata:
    def test_input_types(self):
        inputs = ZhiYiQwenDetectNode.INPUT_TYPES()
        required = inputs["required"]
        assert "image" in required
        assert "target" in required
        assert "model" in required
        assert "score_threshold" in required
        assert "bbox_selection" in required
        assert "merge_boxes" in required

    def test_coordinate_format_in_optional(self):
        inputs = ZhiYiQwenDetectNode.INPUT_TYPES()
        optional = inputs.get("optional", {})
        assert "coordinate_format" in optional

    def test_return_types(self):
        assert ZhiYiQwenDetectNode.RETURN_TYPES == ("JSON", "BBOX", "BBOXES")
        assert ZhiYiQwenDetectNode.RETURN_NAMES == ("json", "bboxes", "bboxes_for_sam2")

    def test_category(self):
        assert ZhiYiQwenDetectNode.CATEGORY == "知衣/目标检测"

    def test_node_switch_returns_empty(self):
        node = ZhiYiQwenDetectNode()
        result = node.detect(
            image="fake",
            target="test",
            node_switch=1,
        )
        assert result == ("[]", [], [[]])

    def test_detect_uploads_image_and_sends_oss_url(self, monkeypatch):
        node = ZhiYiQwenDetectNode()
        image = torch.zeros((1, 10, 20, 3), dtype=torch.float32)
        upload_calls = []
        post_calls = []

        def fake_upload_images(image_tensors):
            upload_calls.append(image_tensors)
            return ["https://oss.example.com/qwen-input.png"]

        class DummyResponse:
            ok = True
            status_code = 200
            text = ""

            def json(self):
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": '[{"bbox_2d":[100,200,500,800],"label":"bag","score":0.9}]',
                            },
                        }
                    ]
                }

        def fake_post(url, headers, data, timeout):
            post_calls.append({
                "url": url,
                "headers": headers,
                "payload": json.loads(data),
                "timeout": timeout,
            })
            return DummyResponse()

        monkeypatch.setattr(node.gemini_client, "upload_images", fake_upload_images)
        monkeypatch.setattr(qwen_module.requests, "post", fake_post)

        json_output, bboxes, bboxes_for_sam2 = node.detect(
            image=image,
            target="bag",
            model="qwen3.6-flash",
            score_threshold=0.0,
            bbox_selection="all",
            merge_boxes=False,
            coordinate_format="1000",
        )

        assert len(upload_calls) == 1
        assert upload_calls[0][0].shape == (1, 10, 20, 3)
        image_part = post_calls[0]["payload"]["messages"][1]["content"][0]
        assert image_part == {
            "type": "image_url",
            "image_url": {"url": "https://oss.example.com/qwen-input.png"},
        }
        assert not image_part["image_url"]["url"].startswith("data:image/")
        assert json.loads(json_output) == [{"bbox_2d": [2, 2, 10, 8], "label": "bag"}]
        assert bboxes == [[2, 2, 10, 8]]
        assert bboxes_for_sam2 == [[[2, 2, 10, 8]]]


def _json_response(status_code: int, body: dict[str, object]) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode("utf-8")
    return response


DetectionMocks = tuple[ZhiYiQwenDetectNode, Mock, Mock, Mock]


@pytest.fixture
def detection_mocks(monkeypatch: pytest.MonkeyPatch) -> DetectionMocks:
    node = ZhiYiQwenDetectNode()
    upload = Mock(return_value=["https://oss.example.com/qwen-input.png"])
    post = Mock(return_value=_json_response(200, {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": '[{"bbox_2d":[100,200,500,800],"label":"bag","score":0.9}]'},
        }],
    }))
    sleep = Mock()
    monkeypatch.setattr(node.gemini_client, "upload_images", upload)
    monkeypatch.setattr(qwen_module.requests, "post", post)
    monkeypatch.setattr(qwen_module.time, "sleep", sleep)
    return node, upload, post, sleep


def test_detect_retries_upstream_image_download_timeout(detection_mocks: DetectionMocks) -> None:
    node, upload, post, sleep = detection_mocks
    failure = _json_response(400, {"error": {
        "message": "litellm.BadRequestError: OpenAIException - Timeout while downloading "
                   "url=https://oss.example.com/qwen-input.png Request id: test-request-id",
        "type": "BadRequest",
        "param": "image_url",
        "code": "400",
    }})
    post.side_effect = [failure, failure, post.return_value]

    result = node.detect(torch.zeros((1, 10, 20, 3)), "bag", model="doubao-seed-2-0-mini")

    assert json.loads(result[0]) == [{"bbox_2d": [2, 2, 10, 8], "label": "bag"}]
    assert result[1:] == ([[2, 2, 10, 8]], [[[2, 2, 10, 8]]])
    upload.assert_called_once()
    assert post.call_count == 3
    assert post.call_args_list == [post.call_args_list[0]] * 3
    payload = json.loads(post.call_args.kwargs["data"])
    assert payload["model"] == "doubao-seed-2-0-mini"
    assert payload["messages"][1]["content"][0]["image_url"]["url"] == "https://oss.example.com/qwen-input.png"
    assert sleep.call_args_list == [call(1), call(2)]


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_detect_retries_transient_http_errors(detection_mocks: DetectionMocks, status_code: int) -> None:
    node, _, post, sleep = detection_mocks
    post.side_effect = [_json_response(status_code, {"error": "temporarily unavailable"}), post.return_value]

    assert node.detect(torch.zeros((1, 10, 20, 3)), "bag")[1] == [[2, 2, 10, 8]]
    assert post.call_count == 2
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize("error_type", [requests.exceptions.Timeout, requests.exceptions.ConnectionError])
def test_detect_retries_network_errors(detection_mocks: DetectionMocks, error_type: type[Exception]) -> None:
    node, _, post, sleep = detection_mocks
    post.side_effect = [error_type("temporary network failure"), post.return_value]

    assert node.detect(torch.zeros((1, 10, 20, 3)), "bag")[1] == [[2, 2, 10, 8]]
    assert post.call_count == 2
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize("failure_kind", ["http", "timeout", "connection"])
def test_detect_stops_after_retry_budget_and_preserves_last_error(detection_mocks: DetectionMocks, failure_kind: str) -> None:
    node, upload, post, sleep = detection_mocks
    if failure_kind == "http":
        post.side_effect = [
            _json_response(400, {"error": f"Timeout while downloading url=test.png Request id: request-{i}"})
            for i in range(3)
        ]
        expected_error = "request-2"
    else:
        error_type = requests.exceptions.Timeout if failure_kind == "timeout" else requests.exceptions.ConnectionError
        post.side_effect = [error_type(f"network failure {i}") for i in range(3)]
        expected_error = "network failure 2"

    with pytest.raises(RuntimeError, match=expected_error) as exc_info:
        node.detect(torch.zeros((1, 10, 20, 3)), "bag")

    if failure_kind == "timeout":
        assert str(exc_info.value).startswith("TIMEOUT:")
    assert post.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    upload.assert_called_once()


@pytest.mark.parametrize(("status_code", "message"), [
    (400, "invalid image_url"),
    (400, "unsupported model parameter"),
    (401, "Timeout while downloading url=test.png"),
    (403, "permission denied"),
    (404, "model not found"),
])
def test_detect_does_not_retry_permanent_http_errors(detection_mocks: DetectionMocks, status_code: int, message: str) -> None:
    node, _, post, sleep = detection_mocks
    post.return_value = _json_response(status_code, {"error": message})

    with pytest.raises(RuntimeError, match=str(status_code)):
        node.detect(torch.zeros((1, 10, 20, 3)), "bag")

    post.assert_called_once()
    sleep.assert_not_called()


def test_detect_can_disable_retries(detection_mocks: DetectionMocks) -> None:
    node, _, post, sleep = detection_mocks
    post.return_value = _json_response(503, {"error": "service unavailable"})

    with pytest.raises(RuntimeError, match="503"):
        node.detect(torch.zeros((1, 10, 20, 3)), "bag", retry_count=0)

    post.assert_called_once()
    sleep.assert_not_called()


def test_detect_caps_retry_count(detection_mocks: DetectionMocks) -> None:
    node, _, post, sleep = detection_mocks
    post.return_value = _json_response(503, {"error": "service unavailable"})

    with pytest.raises(RuntimeError, match="503"):
        node.detect(torch.zeros((1, 10, 20, 3)), "bag", retry_count=10)

    assert post.call_count == 6
    assert sleep.call_args_list == [call(1), call(2), call(2), call(2), call(2)]


@pytest.mark.parametrize("invalid_json", [True, False])
def test_detect_does_not_retry_response_parsing_errors(detection_mocks: DetectionMocks, invalid_json: bool) -> None:
    node, _, post, sleep = detection_mocks
    post.return_value = _json_response(200, {"unexpected": "response"})
    if invalid_json:
        post.return_value._content = b"invalid JSON"

    with pytest.raises(RuntimeError):
        node.detect(torch.zeros((1, 10, 20, 3)), "bag")

    post.assert_called_once()
    sleep.assert_not_called()
