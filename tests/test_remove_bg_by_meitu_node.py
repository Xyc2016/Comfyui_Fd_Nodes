import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import remove_bg_by_meitu_node as remove_bg_module
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.remove_bg_by_meitu_node import ZhiYiRemoveBgByMeituNode


def test_remove_bg_node_metadata():
    input_types = ZhiYiRemoveBgByMeituNode.INPUT_TYPES()

    assert input_types["required"]["images"][0] == "IMAGE"
    assert input_types["required"]["service_url"][1]["default"].endswith("/image/remove_bg_by_meitu")
    assert input_types["required"]["background_color"][0] == "STRING"
    assert "model" not in input_types["required"]
    assert ZhiYiRemoveBgByMeituNode.RETURN_TYPES == ("IMAGE", "MASK", "IMAGE", "IMAGE")
    assert ZhiYiRemoveBgByMeituNode.RETURN_NAMES == ("IMAGE", "MASK", "MASK_IMAGE", "RED_EDGE_IMAGE")
    assert ZhiYiRemoveBgByMeituNode.CATEGORY == "知衣/抠图"
    assert NODE_CLASS_MAPPINGS["ZhiYiRemoveBgByMeituNode"] is ZhiYiRemoveBgByMeituNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiRemoveBgByMeituNode"] == "知衣-美图服装抠图"


def test_extract_urls_accepts_direct_and_wrapped_response():
    node = ZhiYiRemoveBgByMeituNode()

    direct = node._extract_urls({
        "status": True,
        "result_url": "https://example.com/result.png",
        "main_mask_url": "https://example.com/mask.png",
        "red_edge_image_url": "https://example.com/red.png",
    })
    wrapped = node._extract_urls({
        "data": {
            "status": True,
            "result_url": "https://example.com/result-2.png",
            "main_mask_url": "https://example.com/mask-2.png",
            "red_edge_image_url": "https://example.com/red-2.png",
        }
    })

    assert direct == {
        "result_url": "https://example.com/result.png",
        "main_mask_url": "https://example.com/mask.png",
        "red_edge_image_url": "https://example.com/red.png",
    }
    assert wrapped["result_url"] == "https://example.com/result-2.png"
    assert wrapped["main_mask_url"] == "https://example.com/mask-2.png"


def test_extract_urls_uses_error_message_for_failed_status():
    node = ZhiYiRemoveBgByMeituNode()

    with pytest.raises(RuntimeError, match="E123: bad mask"):
        node._extract_urls({
            "status": False,
            "error": {
                "code": "E123",
                "message": "bad mask",
            },
        })


def test_extract_urls_requires_result_and_mask_urls():
    node = ZhiYiRemoveBgByMeituNode()

    with pytest.raises(RuntimeError, match="result_url"):
        node._extract_urls({"status": True, "main_mask_url": "https://example.com/mask.png"})
    with pytest.raises(RuntimeError, match="main_mask_url"):
        node._extract_urls({"status": True, "result_url": "https://example.com/result.png"})


def test_process_mask_image_and_mask_visualization_keep_shapes():
    node = ZhiYiRemoveBgByMeituNode()
    image = Image.new("L", (4, 3), 0)
    image.putpixel((1, 1), 255)

    mask = node._process_mask_image(image, mask_blur=1, mask_offset=1, invert_output=False)
    inverted = node._process_mask_image(image, mask_blur=0, mask_offset=0, invert_output=True)
    mask_image = node._mask_to_image_tensor(mask)

    assert mask.shape == (1, 3, 4)
    assert mask_image.shape == (1, 3, 4, 3)
    assert torch.all(mask >= 0)
    assert torch.all(mask <= 1)
    assert inverted[0, 1, 1] == 0
    assert inverted[0, 0, 0] == 1


def test_compose_color_background_uses_mask_and_hex_color():
    node = ZhiYiRemoveBgByMeituNode()
    original = torch.ones((1, 2, 2, 3), dtype=torch.float32)
    mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    result = node._compose_color_background(original, mask, "#000000")

    assert result.shape == (1, 2, 2, 3)
    assert torch.equal(result[0, 0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(result[0, 0, 1], torch.tensor([0.0, 0.0, 0.0]))


def test_request_with_retry_retries_timeout_once(monkeypatch):
    node = ZhiYiRemoveBgByMeituNode()
    calls = []

    class DummyResponse:
        status_code = 200
        content = b"ok"

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url, timeout, kwargs))
        if len(calls) == 1:
            raise remove_bg_module.requests.exceptions.Timeout("timeout")
        return DummyResponse()

    monkeypatch.setattr(remove_bg_module.requests, "request", fake_request)

    response = node._request_with_retry("GET", "https://example.com/image.png", timeout=3, retry_count=1)

    assert response.status_code == 200
    assert len(calls) == 2


def test_run_concurrent_keeps_result_slots_by_input_index():
    node = ZhiYiRemoveBgByMeituNode()

    def task(value):
        return {
            "result_image": value,
            "mask": value,
            "mask_image": value,
            "red_edge_image": value,
            "image_url": f"input-{value}",
            "result_url": f"result-{value}",
            "main_mask_url": f"mask-{value}",
        }

    tasks = [
        (0, task, ("a",)),
        (1, task, ("b",)),
        (2, task, ("c",)),
    ]

    results, log_lines, last_error = node._run_concurrent(tasks, max_workers=3)

    assert [result["result_image"] for result in results] == ["a", "b", "c"]
    assert len(log_lines) == 3
    assert last_error is None


def test_remove_bg_returns_successful_batch_and_skips_failures(monkeypatch):
    node = ZhiYiRemoveBgByMeituNode()
    images = torch.zeros((3, 2, 2, 3), dtype=torch.float32)

    def fake_single_request(
        idx,
        image_tensor,
        service_url,
        repaint_edge,
        edge_thickness,
        mask_blur,
        mask_offset,
        invert_output,
        background,
        background_color,
        timeout,
    ):
        if idx == 1:
            raise RuntimeError("failed")
        value = float(idx + 1)
        return {
            "result_image": torch.full((1, 2, 2, 3), value),
            "mask": torch.full((1, 2, 2), value),
            "mask_image": torch.full((1, 2, 2, 3), value),
            "red_edge_image": torch.full((1, 2, 2, 3), value),
            "image_url": f"https://example.com/input-{idx}.png",
            "result_url": f"https://example.com/result-{idx}.png",
            "main_mask_url": f"https://example.com/mask-{idx}.png",
        }

    monkeypatch.setattr(node, "_single_request", fake_single_request)

    result_image, mask, mask_image, red_edge = node.remove_bg(
        images=images,
        service_url="https://example.com/image/remove_bg_by_meitu",
        repaint_edge=True,
        edge_thickness=40,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        background="Alpha",
        background_color="#222222",
        max_concurrency=3,
        timeout=30,
    )

    assert result_image.shape == (2, 2, 2, 3)
    assert mask.shape == (2, 2, 2)
    assert mask_image.shape == (2, 2, 2, 3)
    assert red_edge.shape == (2, 2, 2, 3)
    assert torch.equal(result_image[:, 0, 0, 0], torch.tensor([1.0, 3.0]))


def test_remove_bg_requires_service_url_when_no_default_is_available(monkeypatch):
    node = ZhiYiRemoveBgByMeituNode()
    monkeypatch.setattr(remove_bg_module, "FD_REMOVE_BG_BY_MEITU_URL", "")

    with pytest.raises(RuntimeError, match="FD_REMOVE_BG_BY_MEITU_URL"):
        node.remove_bg(
            images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            service_url="",
            repaint_edge=True,
            edge_thickness=40,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            background="Alpha",
            background_color="#222222",
            max_concurrency=1,
            timeout=30,
        )


def test_remove_bg_uses_default_service_url_when_input_is_blank(monkeypatch):
    node = ZhiYiRemoveBgByMeituNode()
    images = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    captured = []
    monkeypatch.setattr(
        remove_bg_module,
        "FD_REMOVE_BG_BY_MEITU_URL",
        "http://image-server-internal.zhiyi.com.cn/api-server-gray/detail-image/image/remove_bg_by_meitu",
    )

    def fake_run_concurrent(tasks, max_workers):
        idx, fn, args = tasks[0]
        captured.append((idx, fn, args, max_workers))
        return (
            [{
                "result_image": torch.ones((1, 2, 2, 3), dtype=torch.float32),
                "mask": torch.ones((1, 2, 2), dtype=torch.float32),
                "mask_image": torch.ones((1, 2, 2, 3), dtype=torch.float32),
                "red_edge_image": torch.ones((1, 2, 2, 3), dtype=torch.float32),
            }],
            ["ok"],
            None,
        )

    monkeypatch.setattr(node, "_run_concurrent", fake_run_concurrent)

    node.remove_bg(
        images=images,
        service_url="",
        repaint_edge=True,
        edge_thickness=40,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        background="Alpha",
        background_color="#222222",
        max_concurrency=1,
        timeout=30,
    )

    assert captured[0][2][2] == "http://image-server-internal.zhiyi.com.cn/api-server-gray/detail-image/image/remove_bg_by_meitu"
