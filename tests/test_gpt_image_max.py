"""Local HTTP boundaries and controlled timing; never call a model supplier."""

import base64
import io
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import aistudio_image_combo_node as node_module
from src.Comfyui_Fd_Nodes.aistudio_image_combo_node import ZhiYiAiStudioImageComboNode
from src.Comfyui_Fd_Nodes.utils.gpt_image_max import (
    MAX_MODELS,
    call_with_same_model_overlap,
    request_max_image_edit,
)


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(buffer, format="PNG")
    return buffer.getvalue()


@contextmanager
def server(payload: dict, *, status: int = 200, download_status: int = 200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            calls.append((self.path, self.headers, self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(payload).replace("IMAGE_URL", self.server.base_url + "/image.png").encode()
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            calls.append((self.path, self.headers, b""))
            self.send_response(download_status)
            self.end_headers()
            self.wfile.write(png())

        def log_message(self, *args: object) -> None:
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.base_url = f"http://127.0.0.1:{http.server_port}"
    worker = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield http.base_url, calls
    finally:
        http.shutdown()
        worker.join(timeout=2)
        http.server_close()


@pytest.mark.parametrize("model", sorted(MAX_MODELS))
@pytest.mark.parametrize("kind", ["url", "base64"])
def test_max_node_posts_exact_model_and_decodes_real_http(model: str, kind: str) -> None:
    item = {"url": "IMAGE_URL", "b64_json": None} if kind == "url" else {"b64_json": base64.b64encode(png()).decode()}
    with server({"data": [item]}) as (base_url, calls):
        node = ZhiYiAiStudioImageComboNode()
        assert node._resolve_api_type("auto", model) == "gpt_image"
        output, task_id, url = node._single_gpt_request(
            base_url=base_url, api_key="internal-test", model=model, prompt="red cup",
            images=[torch.zeros((1, 3, 4, 3))], aspect_ratio="3:4", image_size="2K",
            quality="max", seed=1, out_request_id="xyc_dev_max", size_override="1536x2048",
        )
        assert tuple(output.shape) == (1, 3, 4, 3)
        assert torch.all(output[..., 0] == 1) and torch.all(output[..., 1:] == 0)
        assert task_id is None
        assert bool(url) == (kind == "url")
        assert calls[0][0] == "/v1/images/edits"
        assert calls[0][1]["Authorization"] == "Bearer internal-test"
        body = calls[0][2]
        for field, value in [("model", model), ("size", "1536x2048"), ("quality", "max"), ("user", "xyc_dev_max")]:
            assert f'name="{field}"\r\n\r\n{value}'.encode() in body
        assert b'filename="image_0.png"' in body
        assert len(calls) == (2 if kind == "url" else 1)


@pytest.mark.parametrize("payload", [{"data": []}, {"data": [{"url": None, "b64_json": None}]}])
def test_max_rejects_missing_image(payload: dict) -> None:
    with server(payload) as (url, _):
        with pytest.raises(ValueError, match="no image data|neither"):
            request_max_image_edit(base_url=url, api_key="internal", data={}, multipart_files=[], timeout=1)


def test_download_http_error_is_not_success() -> None:
    with server({"data": [{"url": "IMAGE_URL"}]}, download_status=404) as (url, _):
        with pytest.raises(requests.HTTPError):
            request_max_image_edit(base_url=url, api_key="internal", data={}, multipart_files=[], timeout=1)


@pytest.mark.parametrize("status,payload", [
    (401, {"error": {"message": "Invalid token"}}),
    (400, {"error": {"message": "invalid size"}}),
    (500, {"error": {"message": "safety content_filter"}}),
    (200, {"error": {"code": "NSFW", "message": "blocked"}}),
])
def test_terminal_errors_never_retry_or_fallback(status: int, payload: dict) -> None:
    with server(payload, status=status) as (url, calls):
        with pytest.raises((requests.HTTPError, RuntimeError)):
            call_with_same_model_overlap(
                lambda timeout: request_max_image_edit(
                    base_url=url, api_key="internal", data={"model": sorted(MAX_MODELS)[0]},
                    multipart_files=[], timeout=timeout,
                ), wait_seconds=0.5, total_seconds=1,
            )
        assert len(calls) == 1


def test_fast_success_only_calls_once() -> None:
    calls = []
    result = call_with_same_model_overlap(lambda timeout: calls.append(timeout) or "first", wait_seconds=0.1, total_seconds=0.4)
    assert result == "first" and len(calls) == 1


@pytest.mark.parametrize("winner", [1, 2])
def test_overlap_keeps_first_in_flight_and_does_not_join_loser(winner: int) -> None:
    calls = []
    release = threading.Event()
    second_started = threading.Event()

    def call(timeout: float) -> int:
        index = len(calls) + 1
        calls.append(timeout)
        if index == 2:
            second_started.set()
        assert second_started.wait(0.5)
        if index != winner:
            release.wait(1)
        return index

    try:
        started = time.monotonic()
        assert call_with_same_model_overlap(call, wait_seconds=0.03, total_seconds=0.4) == winner
        assert time.monotonic() - started < 0.3
        assert len(calls) == 2 and 0 < calls[1] < calls[0] <= 0.4
    finally:
        release.set()


def test_retryable_error_starts_second_early() -> None:
    calls = []

    def call(timeout: float) -> str:
        calls.append(timeout)
        if len(calls) == 1:
            raise requests.ConnectionError("connection reset")
        return "second"

    started = time.monotonic()
    assert call_with_same_model_overlap(call, wait_seconds=1, total_seconds=2) == "second"
    assert len(calls) == 2 and time.monotonic() - started < 0.5


def test_two_failures_stop_without_third_attempt() -> None:
    calls = []

    def call(timeout: float) -> str:
        calls.append(timeout)
        raise requests.Timeout("upstream timed out")

    with pytest.raises(requests.Timeout, match="upstream"):
        call_with_same_model_overlap(call, wait_seconds=0.1, total_seconds=1)
    assert len(calls) == 2


def test_total_budget_exits_without_waiting_for_threads() -> None:
    release = threading.Event()
    calls = []

    def call(timeout: float) -> str:
        calls.append(timeout)
        release.wait(1)
        return "too late"

    try:
        with pytest.raises(requests.Timeout, match="total budget"):
            call_with_same_model_overlap(call, wait_seconds=0.02, total_seconds=0.06)
        assert len(calls) == 2
    finally:
        release.set()


@pytest.mark.parametrize("model", sorted(MAX_MODELS))
def test_generate_uses_existing_custom_or_environment_credentials(monkeypatch, model: str) -> None:
    node = ZhiYiAiStudioImageComboNode()
    captured = []
    monkeypatch.setattr(node_module, "FD_LITELLM_BASE_URL", "http://internal-litellm")
    monkeypatch.setattr(node_module, "FD_LITELLM_API_KEY", "internal")
    monkeypatch.setattr(node, "_run_concurrent", lambda tasks, *args, **kwargs: (captured.extend(tasks) or ["image"], [], None))
    for address, key, expected_address, expected_key in [
        ("", "", "http://internal-litellm", "internal"),
        ("http://custom/v1/images/edits", "custom", "http://custom", "custom"),
    ]:
        node.generate(model=model, aspect_ratio="3:4", image_size="2K", batch_size=1,
                      max_concurrency=1, seed_mode="固定种子", seed=1, target_url=address, api_key=key,
                      combo_1={"images": [torch.zeros((1, 3, 4, 3))], "prompts": ["test"]})
        assert captured[-1][1] == node._single_gpt_request
        assert captured[-1][2][:3] == (expected_address, expected_key, model)
