import pytest
import torch

from src.Comfyui_Fd_Nodes import aistudio_image_combo_node as aistudio_image_combo_module
from src.Comfyui_Fd_Nodes.aistudio_image_combo_node import ZhiYiAiStudioImageComboNode


def test_aistudio_combo_node_metadata():
    input_types = ZhiYiAiStudioImageComboNode.INPUT_TYPES()

    assert input_types["required"]["model"][0] == "STRING"
    assert input_types["required"]["model"][1]["default"] == "nano-banana-pro"
    assert input_types["required"]["quality"][0] == ["low", "medium", "high"]
    assert "combo_1" in input_types["optional"]
    assert ZhiYiAiStudioImageComboNode.RETURN_TYPES == ("IMAGE", "INT", "STRING")
    assert ZhiYiAiStudioImageComboNode.OUTPUT_IS_LIST == (True, False, False)
    assert {"base_url", "api_key"}.isdisjoint(input_types["required"])


def test_aistudio_combo_payload_normalizes_image_size_to_uppercase_k():
    node = ZhiYiAiStudioImageComboNode()

    payload = node._build_payload("prompt", ["https://example.com/input.png"], "1:1", "2k")

    assert payload["payload"]["image_size"] == "2K"


def test_aistudio_combo_single_request_posts_publish_payload(monkeypatch):
    node = ZhiYiAiStudioImageComboNode()
    captured = []

    class DummyPostResponse:
        ok = True
        status_code = 200
        text = '{"taskId":"task-1","success":true,"data":{"url":"https://example.com/result.png"}}'

        def json(self):
            return {
                "taskId": "task-1",
                "success": True,
                "data": {"url": "https://example.com/result.png"},
            }

    def fake_post(url, headers, json, timeout):
        captured.append(("post", url, headers, json, timeout))
        return DummyPostResponse()

    def fake_download(result_url):
        captured.append(("download", result_url))
        return "fake-image"

    monkeypatch.setattr(aistudio_image_combo_module.requests, "post", fake_post)
    monkeypatch.setattr(node, "_download_result_image", fake_download)

    result = node._single_request(
        publish_url="http://example.com/api/tasks/publish",
        prompt="test prompt",
        image_urls=["https://example.com/input-1.png", "https://example.com/input-2.png"],
        aspect_ratio="3:4",
        image_size="2K",
        model="nano-banana-pro",
        seed=123,
        out_request_id="req-1",
    )

    assert result == ("fake-image", "task-1", "https://example.com/result.png")
    _, url, headers, request_body, timeout = captured[0]
    assert url == "http://example.com/api/tasks/publish"
    assert headers == {"Content-Type": "application/json"}
    assert timeout == 600
    assert request_body == {
        "type": "AiStudio",
        "payload": {
            "prompt": "test prompt",
            "image": ["https://example.com/input-1.png", "https://example.com/input-2.png"],
            "image_size": "2K",
            "aspect_ratio": "3:4",
        },
        "timeout": 300000,
    }
    assert "model" not in request_body["payload"]
    assert "seed" not in request_body["payload"]
    assert "out_request_id" not in request_body["payload"]
    assert captured[1] == ("download", "https://example.com/result.png")


def test_aistudio_combo_generate_reuses_uploaded_urls_and_keeps_combo_shape(monkeypatch):
    node = ZhiYiAiStudioImageComboNode()
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

    result = node.generate(
        model="nano-banana-pro",
        aspect_ratio="",
        image_size="4K",
        quality="medium",
        batch_size=1,
        max_concurrency=2,
        seed_mode="固定种子",
        seed=10,
        out_request_id="req-1",
        combo_1=combo,
        system_prompt="system",
    )

    images, actual_seed, log_text = result
    assert images == ["image-1", "image-2"]
    assert actual_seed == 10
    assert "总计: 2/2 成功" in log_text
    assert len(captured_tasks) == 2
    first_args = captured_tasks[0][2]
    second_args = captured_tasks[1][2]
    assert first_args[1] == "system\n\nprompt one"
    assert first_args[2] == ["https://example.com/input.png"]
    assert first_args[3] == ""
    assert first_args[4] == "4K"
    assert first_args[5] == "nano-banana-pro"
    assert first_args[6] == 10
    assert first_args[7] == "req-1"
    assert second_args[1] == "system\n\nprompt two"
    assert second_args[6] == 11


def test_aistudio_combo_generate_returns_last_actual_error(monkeypatch):
    node = ZhiYiAiStudioImageComboNode()
    combo = {"images": [torch.zeros((1, 2, 2, 3), dtype=torch.float32)], "prompts": ["test prompt"]}

    monkeypatch.setattr(node, "_upload_images", lambda images: ["https://example.com/input.png"])
    monkeypatch.setattr(
        node,
        "_run_concurrent",
        lambda tasks, max_workers, label="任务": ([None] * len(tasks), ["[请求 1] 失败"], "UNKNOWN: API 返回失败: bad request"),
    )

    with pytest.raises(RuntimeError, match=r"^UNKNOWN: API 返回失败"):
        node.generate(
            model="nano-banana-pro",
            aspect_ratio="1:1",
            image_size="4K",
            quality="medium",
            batch_size=1,
            max_concurrency=1,
            combo_1=combo,
        )


def test_aistudio_combo_generate_routes_gpt_image_model_to_litellm_without_upload(monkeypatch):
    node = ZhiYiAiStudioImageComboNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    combo = {"images": [image], "prompts": ["test prompt"]}
    captured_tasks = []

    monkeypatch.setattr(
        aistudio_image_combo_module,
        "FD_LITELLM_BASE_URL",
        "https://litellm.example.com",
    )
    monkeypatch.setattr(aistudio_image_combo_module, "FD_LITELLM_API_KEY", "secret")
    monkeypatch.setattr(node, "_upload_images", lambda images: pytest.fail("GPT Image LiteLLM branch should not upload OSS URLs"))

    def fake_run_concurrent(tasks, max_workers, label="任务"):
        captured_tasks.extend(tasks)
        return (["image-1"], ["[请求 1] 成功"], None)

    monkeypatch.setattr(node, "_run_concurrent", fake_run_concurrent)

    images, actual_seed, log_text = node.generate(
        model="openai/gpt-image-2",
        aspect_ratio="3:4",
        image_size="2K",
        quality="high",
        batch_size=1,
        max_concurrency=1,
        seed_mode="固定种子",
        seed=20,
        out_request_id="req-gpt",
        combo_1=combo,
        system_prompt="system",
    )

    assert images == ["image-1"]
    assert actual_seed == 20
    assert "https://litellm.example.com/v1/images/edits" in log_text
    assert len(captured_tasks) == 1
    task_idx, fn, args = captured_tasks[0]
    assert task_idx == 0
    assert fn == node._single_gpt_request
    assert args[0] == "https://litellm.example.com"
    assert args[1] == "secret"
    assert args[2] == "openai/gpt-image-2"
    assert args[3] == "system\n\ntest prompt"
    assert len(args[4]) == 1
    assert torch.equal(args[4][0], image)
    assert args[5] == "3:4"
    assert args[6] == "2K"
    assert args[7] == "high"
    assert args[8] == 20
    assert args[9] == "req-gpt"


def test_aistudio_combo_single_gpt_request_uses_gpt_edits_payload(monkeypatch):
    node = ZhiYiAiStudioImageComboNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return aistudio_image_combo_module.BytesIO(node._tensor_to_png_bytes(image)), "text", "https://example.com/result.png"

    monkeypatch.setattr(node, "_call_gpt_image_with_retry_policy", fake_call)

    result_image, task_id, result_url = node._single_gpt_request(
        base_url="https://litellm.example.com",
        api_key="secret",
        model="gpt-image-2",
        prompt="test prompt",
        images=[image],
        aspect_ratio="9:16",
        image_size="1K",
        quality="LOW",
        seed=1,
        out_request_id="req-1",
    )

    assert tuple(result_image.shape) == (1, 2, 2, 3)
    assert task_id is None
    assert result_url == "https://example.com/result.png"
    assert captured["base_url"] == "https://litellm.example.com"
    assert captured["api_key"] == "secret"
    assert captured["data"] == {
        "model": "gpt-image-2",
        "prompt": "test prompt",
        "size": "720x1280",
        "quality": "low",
        "user": "req-1",
    }
    assert len(captured["multipart_files"]) == 1
    assert captured["multipart_files"][0][0] == "image"
