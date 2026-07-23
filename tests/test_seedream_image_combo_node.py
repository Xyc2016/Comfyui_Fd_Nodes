import pytest
import torch

from src.Comfyui_Fd_Nodes import seedream_image_combo_node as seedream_combo_module
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.seedream_image_combo_node import FD_SeedreamImageComboNode


def test_seedream_combo_node_metadata():
    input_types = FD_SeedreamImageComboNode.INPUT_TYPES()

    assert input_types["required"]["model"][0] == ["doubao-seedream-5.0-lite", "doubao-seedream-5.0-pro"]
    assert input_types["required"]["size"][0] == ["4K", "3K", "2K", "1K"]
    assert input_types["required"]["size"][1]["default"] == "2K"
    assert input_types["required"]["output_format"][0] == ["png", "jpg"]
    assert "combo_1" in input_types["optional"]
    assert "combo_8" in input_types["optional"]
    assert "aspect_ratio" not in input_types["required"]
    assert input_types["optional"]["aspect_ratio"][0] == ["1:1", "3:4", "4:3", "16:9", "9:16", "3:2", "2:3", "21:9"]
    assert FD_SeedreamImageComboNode.RETURN_TYPES == ("IMAGE", "INT", "STRING")
    assert FD_SeedreamImageComboNode.RETURN_NAMES == ("image", "seed", "log")
    assert FD_SeedreamImageComboNode.OUTPUT_IS_LIST == (True, False, False)
    assert NODE_CLASS_MAPPINGS["FD_SeedreamImageComboNode"] is FD_SeedreamImageComboNode
    assert NODE_DISPLAY_NAME_MAPPINGS["FD_SeedreamImageComboNode"] == "FD Seedream Image Combo"


def test_seedream_combo_single_request_posts_generation_payload(monkeypatch):
    node = FD_SeedreamImageComboNode()
    captured = []

    class DummyPostResponse:
        ok = True
        status_code = 200
        text = '{"data":[{"url":"https://example.com/result.png"}]}'

        def json(self):
            return {"data": [{"url": "https://example.com/result.png"}]}

    def fake_post(url, headers, json, timeout):
        captured.append(("post", url, headers, json, timeout))
        return DummyPostResponse()

    def fake_download(result_url):
        captured.append(("download", result_url))
        return "fake-image"

    monkeypatch.setattr(seedream_combo_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_download_result_image", fake_download)

    result = node._single_request(
        base_url="https://example.com",
        api_key="secret",
        model="doubao-seedream-5.0-lite",
        prompt="test prompt",
        image_urls=["https://example.com/input-1.png", "https://example.com/input-2.png"],
        size="2K",
        output_format="png",
        seed=123,
        aspect_ratio="3:4",
    )

    assert result == ("fake-image", None, "https://example.com/result.png")
    _, url, headers, request_body, timeout = captured[0]
    assert url == "https://example.com/v1/images/generations"
    assert headers == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }
    assert timeout == 300
    assert request_body == {
        "model": "doubao-seedream-5.0-lite",
        "prompt": "test prompt",
        "sequential_image_generation": "disabled",
        "size": "1728x2304",
        "output_format": "png",
        "watermark": False,
        "image": ["https://example.com/input-1.png", "https://example.com/input-2.png"],
    }
    assert "seed" not in request_body
    assert "aspect_ratio" not in request_body
    assert "ratio" not in request_body
    assert captured[1] == ("download", "https://example.com/result.png")


def test_seedream_combo_single_request_omits_sequential_image_generation_for_pro(monkeypatch):
    node = FD_SeedreamImageComboNode()
    captured = []

    class DummyPostResponse:
        ok = True
        status_code = 200
        text = '{"data":[{"url":"https://example.com/result.png"}]}'

        def json(self):
            return {"data": [{"url": "https://example.com/result.png"}]}

    def fake_post(url, headers, json, timeout):
        captured.append(("post", url, headers, json, timeout))
        return DummyPostResponse()

    monkeypatch.setattr(seedream_combo_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_download_result_image", lambda result_url: "fake-image")

    result = node._single_request(
        base_url="https://example.com",
        api_key="secret",
        model="doubao-seedream-5.0-pro",
        prompt="test prompt",
        image_urls=["https://example.com/input.png"],
        size="2K",
        output_format="png",
        seed=123,
        aspect_ratio="1:1",
    )

    assert result == ("fake-image", None, "https://example.com/result.png")
    request_body = captured[0][3]
    assert request_body == {
        "model": "doubao-seedream-5.0-pro",
        "prompt": "test prompt",
        "size": "2048x2048",
        "output_format": "png",
        "watermark": False,
        "image": ["https://example.com/input.png"],
    }
    assert "sequential_image_generation" not in request_body
    assert "aspect_ratio" not in request_body
    assert "ratio" not in request_body


@pytest.mark.parametrize(
    ("model", "has_sequential_field"),
    [
        ("doubao-seedream-5.0-lite", True),
        ("doubao-seedream-5.0-pro", False),
    ],
)
def test_seedream_combo_builds_1k_payload_for_lite_and_pro(model, has_sequential_field):
    node = FD_SeedreamImageComboNode()
    image_urls = ["https://example.com/input-1.png", "https://example.com/input-2.png"]

    body = node._build_body(model, "test prompt", image_urls, "1K", "png", "1:1")

    assert body["size"] == "1024x1024"
    assert body["image"] == image_urls
    assert ("sequential_image_generation" in body) is has_sequential_field


def test_seedream_combo_generate_reuses_uploaded_urls_and_keeps_combo_shape(monkeypatch):
    node = FD_SeedreamImageComboNode()
    combo = {
        "images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "prompts": ["prompt one", "prompt two"],
    }
    captured_tasks = []

    monkeypatch.setattr(node, "_upload_images", lambda images: ["https://example.com/input.png"])

    def fake_run_concurrent(tasks, max_workers, label="任务"):
        captured_tasks.extend(tasks)
        return (["image-1", "image-2"], ["[请求 1] 成功", "[请求 2] 成功"], None)

    monkeypatch.setattr(node, "_run_concurrent", fake_run_concurrent)

    images, actual_seed, log_text = node.generate(
        model="doubao-seedream-5.0-lite",
        size="3K",
        output_format="jpg",
        batch_size=1,
        max_concurrency=2,
        seed_mode="固定种子",
        seed=10,
        combo_1=combo,
        system_prompt="system",
        aspect_ratio="16:9",
    )

    assert images == ["image-1", "image-2"]
    assert actual_seed == 10
    assert "总计: 2/2 成功" in log_text
    assert len(captured_tasks) == 2
    first_args = captured_tasks[0][2]
    second_args = captured_tasks[1][2]
    assert first_args[0] == "https://example.com"
    assert first_args[1] == "test-key"
    assert first_args[2] == "doubao-seedream-5.0-lite"
    assert first_args[3] == "system\n\nprompt one"
    assert first_args[4] == ["https://example.com/input.png"]
    assert first_args[5] == "3K"
    assert first_args[6] == "jpg"
    assert first_args[7] == 10
    assert first_args[8] == "16:9"
    assert second_args[3] == "system\n\nprompt two"
    assert second_args[7] == 11
    assert second_args[8] == "16:9"


def test_seedream_combo_generate_rejects_missing_combo():
    node = FD_SeedreamImageComboNode()

    with pytest.raises(RuntimeError, match="未提供任何组合输入"):
        node.generate(
            model="doubao-seedream-5.0-lite",
            size="2K",
            output_format="png",
            batch_size=1,
            max_concurrency=1,
        )


def test_seedream_combo_generate_returns_preprocess_error_when_no_valid_tasks():
    node = FD_SeedreamImageComboNode()

    with pytest.raises(RuntimeError, match="无图片"):
        node.generate(
            model="doubao-seedream-5.0-lite",
            size="2K",
            output_format="png",
            batch_size=1,
            max_concurrency=1,
            combo_1={"images": [], "prompts": ["test prompt"]},
        )


def test_seedream_combo_generate_returns_last_actual_error(monkeypatch):
    node = FD_SeedreamImageComboNode()
    combo = {"images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)], "prompts": ["test prompt"]}

    monkeypatch.setattr(node, "_upload_images", lambda images: ["https://example.com/input.png"])
    monkeypatch.setattr(
        node,
        "_run_concurrent",
        lambda tasks, max_workers, label="任务": ([None] * len(tasks), ["[请求 1] 失败"], "UNKNOWN: API 请求失败: 400"),
    )

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: API 请求失败"):
        node.generate(
            model="doubao-seedream-5.0-lite",
            size="2K",
            output_format="png",
            batch_size=1,
            max_concurrency=1,
            combo_1=combo,
        )


def test_seedream_combo_generate_returns_partial_success_with_failure_log(monkeypatch):
    node = FD_SeedreamImageComboNode()
    combo = {
        "images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)],
        "prompts": ["prompt one", "prompt two"],
    }

    monkeypatch.setattr(node, "_upload_images", lambda images: ["https://example.com/input.png"])
    monkeypatch.setattr(
        node,
        "_run_concurrent",
        lambda tasks, max_workers, label="任务": (
            ["image-1", None],
            ["[请求 1] 成功", "[请求 2] 失败: UNKNOWN: bad request"],
            "UNKNOWN: bad request",
        ),
    )

    images, actual_seed, log_text = node.generate(
        model="doubao-seedream-5.0-lite",
        size="2K",
        output_format="png",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=7,
        combo_1=combo,
    )

    assert images == ["image-1"]
    assert actual_seed == 7
    assert "总计: 1/2 成功" in log_text
    assert "[请求 2] 失败: UNKNOWN: bad request" in log_text
