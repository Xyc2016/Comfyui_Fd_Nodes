import pytest
import torch

from src.Comfyui_Fd_Nodes import seedream_image_combo_node as seedream_combo_module
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.seedream_image_combo_node import FD_SeedreamImageComboNode
from src.Comfyui_Fd_Nodes.utils.seedream_image_client import SeedreamImageClient


class OkResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.text = str(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


def make_fake_client(captured=None, download_content=b"fake-image"):
    def fake_post(url, headers, json, timeout):
        if captured is not None:
            captured.append((url, headers, json, timeout))
        return OkResponse({"status": True, "result_image_url": "https://example.com/result.png"})

    class GetResponse:
        content = download_content

        def raise_for_status(self):
            return None

    return SeedreamImageClient(
        backend="image_generation",
        generate_url="http://image-server/image/generate",
        edit_url="http://image-server/image/edit",
        oss_uploader=lambda path, data: "https://oss.example.com/input.png",
        request_post=fake_post,
        request_get=lambda url, timeout: GetResponse(),
        timeout=60,
    )


def test_seedream_combo_node_metadata():
    input_types = FD_SeedreamImageComboNode.INPUT_TYPES()

    assert input_types["required"]["model"][0] == ["doubao-seedream-5.0-lite", "doubao-seedream-5.0-pro"]
    assert input_types["required"]["size"][0] == ["4K", "3K", "2K", "1K"]
    assert input_types["required"]["size"][1]["default"] == "2K"
    assert input_types["required"]["output_format"][0] == ["png", "jpg"]
    assert "combo_1" in input_types["optional"]
    assert "combo_8" in input_types["optional"]
    assert "aspect_ratio" not in input_types["required"]
    assert input_types["optional"]["aspect_ratio"][0] == [
        "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9", "9:21",
    ]
    assert FD_SeedreamImageComboNode.RETURN_TYPES == ("IMAGE", "INT", "STRING")
    assert FD_SeedreamImageComboNode.RETURN_NAMES == ("image", "seed", "log")
    assert FD_SeedreamImageComboNode.OUTPUT_IS_LIST == (True, False, False)
    assert NODE_CLASS_MAPPINGS["FD_SeedreamImageComboNode"] is FD_SeedreamImageComboNode
    assert NODE_DISPLAY_NAME_MAPPINGS["FD_SeedreamImageComboNode"] == "FD Seedream Image Combo"


def test_seedream_combo_single_request_posts_image_server_payload(monkeypatch):
    node = FD_SeedreamImageComboNode()
    captured = []
    node._client = make_fake_client(captured=captured)
    monkeypatch.setattr(seedream_combo_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    result = node._single_request(
        model="doubao-seedream-5.0-lite",
        prompt="test prompt",
        image_urls=["https://example.com/input-1.png", "https://example.com/input-2.png"],
        size="2K",
        seed=123,
        aspect_ratio="3:4",
    )

    assert result == ("fake-image", None, "https://example.com/result.png")
    url, headers, request_body, timeout = captured[0]
    assert url == "http://image-server/image/edit"
    assert headers == {"Content-Type": "application/json"}
    assert timeout == 60
    assert request_body == {
        "channel": "doubao-seedream-5.0-lite",
        "prompt": "test prompt",
        "size": "2K",
        "ratio": "3:4",
        "resize": True,
        "image_url_list": ["https://example.com/input-1.png", "https://example.com/input-2.png"],
    }
    assert "seed" not in request_body


def test_seedream_combo_single_request_maps_3k_to_2k(monkeypatch):
    node = FD_SeedreamImageComboNode()
    captured = []
    node._client = make_fake_client(captured=captured)
    monkeypatch.setattr(seedream_combo_module, "bytesio_to_image_tensor", lambda _image_bytesio: "fake-image")

    node._single_request(
        model="doubao-seedream-5.0-pro",
        prompt="test prompt",
        image_urls=["https://example.com/input.png"],
        size="3K",
        seed=123,
        aspect_ratio="1:1",
    )

    assert captured[0][3]["size"] == "2K"


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
    assert first_args[0] == "doubao-seedream-5.0-lite"
    assert first_args[1] == "system\n\nprompt one"
    assert first_args[2] == ["https://example.com/input.png"]
    assert first_args[3] == "3K"
    assert first_args[4] == 10
    assert first_args[5] == "16:9"
    assert second_args[1] == "system\n\nprompt two"
    assert second_args[4] == 11
    assert second_args[5] == "16:9"


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
        lambda tasks, max_workers, label="任务": ([None] * len(tasks), ["[请求 1] 失败"], "UNKNOWN: image/edit 失败: bad request"),
    )

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: image/edit 失败"):
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
