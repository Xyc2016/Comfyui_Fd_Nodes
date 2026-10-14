# 知衣-图生图-combo 底层 Image Generation 服务接口文档

本文档只说明 ComfyUI 节点 `知衣-图生图-combo` 主路径背后调用的图片生成服务接口。这个接口适合绕过 ComfyUI 直接调用。

文档绝对路径：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/zhiyi-image-to-image-combo-api.md
```

对应节点代码绝对路径：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/zhiyi_image_to_image_combo_node.py
```

底层服务 client 代码绝对路径：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/gemini_service.py
```

## 1. 结论

`知衣-图生图-combo` 默认不是直接把图片文件传给图片生成服务。它会先把 ComfyUI 图片上传到 OSS，拿到图片 URL，然后调用内部 Image Generation 服务：

```text
POST {FD_GEMINI_URL}
Content-Type: application/json
```

如果你要自己调用这个服务，直接准备可访问的图片 URL，然后按下面的 JSON 结构请求即可。

最小请求体：

```json
{
  "out_request_id": "req-001",
  "prompt": "把图片里的裤子改成老欧美风格牛仔裤，保持人物姿态和背景不变",
  "model": "google/gemini-3-pro-image-preview",
  "aspect_ratio": "1:1",
  "resolution": "4K",
  "image_url_list": [
    "https://your-oss.example.com/input-1.png"
  ]
}
```

成功响应体：

```json
{
  "result_image_url": "https://your-oss.example.com/output.png",
  "message": "ok",
  "cost_time": 18.6
}
```

调用方拿到 `result_image_url` 后，直接下载这张图即可。

## 2. 服务地址

服务地址来自环境变量：

```text
FD_GEMINI_URL
```

配置代码：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/config.py
```

节点运行时实际使用：

```python
GeminiImageServiceClient().service_url
```

也就是：

```text
FD_GEMINI_URL
```

如果没有配置，节点会报错：

```text
未配置 Gemini 服务地址，请设置环境变量 FD_GEMINI_URL
```

http://image-generation.online-server.svc.cluster.local/image/gemini_image 这个是生产地址 ，灰度为http://image-generation.online-server-gray.svc.cluster.local/image/gemini_image

## 3. 请求方法

```http
POST {FD_GEMINI_URL}
Content-Type: application/json
```

当前 `知衣-图生图-combo` 主路径没有在请求里加鉴权 header。它直接：

```python
requests.post(service_url, json=body, timeout=600)
```

如果你的部署环境在网关层增加了鉴权，需要按网关要求额外加 header；这不是当前 ComfyUI 节点代码的一部分。

## 4. 请求字段

### 4.1 完整字段

```json
{
  "out_request_id": "req-001",
  "prompt": "最终提示词",
  "model": "google/gemini-3-pro-image-preview",
  "aspect_ratio": "1:1",
  "resolution": "4K",
  "image_url_list": [
    "https://your-oss.example.com/input-1.png",
    "https://your-oss.example.com/input-2.png"
  ]
}
```

### 4.2 字段说明

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `out_request_id` | string | 否 | 业务请求 ID。节点传空时 client 会用 `"default"`。 |
| `prompt` | string | 是 | 最终提示词。不能为空。 |
| `model` | string | 是 | 模型名。会按节点规则做归一化，详见“模型名映射”。 |
| `aspect_ratio` | string | 否 | 输出比例。可以传空字符串。 |
| `resolution` | string | 否 | 输出尺寸。`知衣-图生图-combo` 会传 `image_size` 到这个字段。 |
| `image_url_list` | array[string] | 是 | 输入图片 URL 列表。不能为空，URL 必须能被图片生成服务访问。 |

### 4.3 `prompt` 拼接规则

在 ComfyUI 节点里，如果配置了 `system_prompt`，节点会先拼接，再发送给服务。

拼接规则：

```text
system_prompt + "\n\n" + prompt
```

如果 `system_prompt` 为空，只发送 `prompt`。

如果你直接调用接口，不需要传 `system_prompt` 字段，直接把拼好的最终提示词放进 `prompt` 即可。

示例：

```json
{
  "prompt": "你是电商商品图修图师，保持主体结构不变。\n\n把图片里的裤子改成老欧美风格牛仔裤"
}
```

### 4.4 `image_url_list`

`image_url_list` 是图片 URL，不是 base64，也不是 multipart 文件。

要求：

```text
1. 必须是非空数组。
2. URL 必须能被图片生成服务访问。
3. 支持多张参考图。
4. 节点侧会把 ComfyUI IMAGE 上传 OSS 后得到这些 URL；你直接调接口时可以用自己的 OSS/CDN URL。
```

节点上传 OSS 的规则：

```text
ComfyUI IMAGE tensor
  -> PNG bytes
  -> 如果总像素超过 3072 * 3072，会等比例缩小
  -> 文件名使用图片 bytes 的 md5
  -> 上传到 {FD_OSS_URL_PATH_PREFIX_GEMINI}/{md5}.png
  -> 拼成 {FD_OSS_URL_PREFIX}{FD_OSS_URL_PATH_PREFIX_GEMINI}/{md5}.png
```

相关环境变量：

```text
FD_OSS_ACCESS_KEY_ID
FD_OSS_ACCESS_KEY_SECRET
FD_OSS_BUCKET_NAME
FD_OSS_ENDPOINT
FD_OSS_URL_PREFIX
FD_OSS_URL_PATH_PREFIX_GEMINI
```

如果你直接调用 Image Generation 服务，可以跳过 ComfyUI 的 OSS 上传逻辑，只要自己传入可访问的图片 URL。

### 4.5 `resolution`

`知衣-图生图-combo` UI 里的 `image_size` 会传给服务端的 `resolution` 字段。

节点可选值：

```text
4K
2K
1080P
720P
```

示例：

```json
{
  "resolution": "4K"
}
```

### 4.6 `aspect_ratio`

`知衣-图生图-combo` 支持的比例：

```text
""
1:1
1:4
1:8
2:3
3:2
3:4
4:1
4:3
4:5
5:4
8:1
9:16
16:9
21:9
```

如果不想指定比例，可以传：

```json
{
  "aspect_ratio": ""
}
```

或在你自己的调用里省略这个字段；但当前节点会传空字符串。

## 5. 模型名映射

节点 UI 允许选择一批模型名，但发送给 Image Generation 服务前，client 会做一次归一化。

归一化逻辑在：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/gemini_service.py
```

映射规则：

| UI / 调用方传入 model | 实际发给 `FD_GEMINI_URL` 的 model |
| --- | --- |
| `google/gemini-3-pro-image-preview` | `google/gemini-3-pro-image-preview` |
| `google/gemini-3.1-flash-image-preview` | `google/gemini-3.1-flash-image-preview` |
| `google/gemini-2.5-flash-image-preview` | `google/gemini-2.5-flash-image-preview` |
| `google/gemini-3-pro-image-preview-official` | `google/gemini-3-pro-image-preview-official` |
| `gemini-3-pro-image-preview` | `google/gemini-3-pro-image-preview` |
| `gemini-3.1-flash-image-preview` | `google/gemini-3.1-flash-image-preview` |
| `gemini-2.5-flash-image-preview` | `google/gemini-2.5-flash-image-preview` |
| `gemini-3-pro-image-preview-official` | `google/gemini-3-pro-image-preview-official` |
| `gemini-3-pro-image-preview-aistudio` | `google/gemini-3-pro-image-preview-official` |
| `gemini-nano-banana-2.1` | `google/gemini-nano-banana-2.1` |
| `gemini-nano-banana-2.1-siphonlab` | `gemini-nano-banana-2.1-siphonlab` |
| `gemini-nano-banana-2.1-wdcloud` | `gemini-nano-banana-2.1-wdcloud` |
| `gemini-nano-banana-2.1-vip` | `gemini-nano-banana-2.1-vip` |
| `batch/gemini-3-pro-image-preview` | `batch/gemini-3-pro-image-preview` |

建议你直接调用服务时使用最终模型名，减少歧义，例如：

```text
google/gemini-3-pro-image-preview
batch/gemini-3-pro-image-preview
google/gemini-3-pro-image-preview-official
google/gemini-nano-banana-2.1
```

`gemini-nano-banana-2.1` 是 image-server 的公开模型名，直连时服务端会接受裸名或 `google/gemini-nano-banana-2.1`。它只支持 `1K`、`2K`、`4K`；节点传入 `720P`/`1080P` 时分别映射为 `1K`/`2K`。编辑场景默认开启色彩偏移修正，关闭时显式传 `enable_color_bias_correction: false`。显式渠道模型 `-siphonlab` / `-wdcloud` / `-vip` 锁定服务端渠道并保持原名发送（带 `google/` 前缀的变体也原样透传）；逆向渠道稳定性弱于官方，建议仅测试或容量兜底使用。


这个模型不会调用 `FD_GEMINI_URL`：

```text
gemini-3-pro-image-preview-siphonlab
```

它在 `知衣-图生图-combo` 中会走 LiteLLM Gemini 图片渠道：

```text
POST {FD_LITELLM_BASE_URL}/v1/chat/completions
```

并把图片转成 base64 放进 `messages`，不走 OSS `image_url_list`。

所以，如果你要复用本文档的 Image Generation 服务接口，不要用 `gemini-3-pro-image-preview-siphonlab`。

## 6. 响应字段

### 6.1 成功响应

```json
{
  "result_image_url": "https://your-oss.example.com/output.png",
  "message": "ok",
  "cost_time": 18.6
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `result_image_url` | string | 是 | 生成结果图 URL。节点会下载这个 URL 并转成 ComfyUI IMAGE。 |
| `message` | string | 否 | 文本消息。新 client 用 `result.get("message", "")`，可以为空。 |
| `cost_time` | number | 否 | 服务端耗时，主要用于日志。 |

节点拿到响应后会做：

```text
GET result_image_url
  -> 下载图片
  -> PIL RGB
  -> torch.Tensor
  -> ComfyUI IMAGE 输出
```

### 6.2 业务错误响应

如果响应 JSON 里包含：

```json
{
  "error": {
    "code": "TIMEOUT",
    "message": "request timed out"
  }
}
```

节点会把它当作失败处理。

当前 client 判断逻辑：

```text
if result.get("error", {}).get("code"):
    raise GeminiImageServiceError(...)
```

### 6.3 HTTP 错误

如果 HTTP status 不是 2xx，节点会报：

```text
API 请求失败: {status_code}
{response_text 前 1000 字符}
```

### 6.4 响应缺字段

如果没有 `result_image_url`，节点会在解析阶段失败。

常见错误：

```text
解析响应失败: 'result_image_url'
```

## 7. curl 示例

### 7.1 单图图生图

```bash
curl -X POST "$FD_GEMINI_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "out_request_id": "req-demo-001",
    "prompt": "把图片里的裤子改成老欧美风格牛仔裤，保持人物姿态、脸部和背景不变",
    "model": "google/gemini-3-pro-image-preview",
    "aspect_ratio": "1:1",
    "resolution": "4K",
    "image_url_list": [
      "https://your-oss.example.com/input.png"
    ]
  }'
```

### 7.2 多参考图

```bash
curl -X POST "$FD_GEMINI_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "out_request_id": "req-demo-002",
    "prompt": "参考第一张的人物，参考第二张的裤子材质，把第一张裤子替换成第二张风格",
    "model": "google/gemini-3-pro-image-preview",
    "aspect_ratio": "3:4",
    "resolution": "4K",
    "image_url_list": [
      "https://your-oss.example.com/person.png",
      "https://your-oss.example.com/reference-pants.png"
    ]
  }'
```

### 7.3 batch 渠道

```bash
curl -X POST "$FD_GEMINI_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "out_request_id": "req-demo-003",
    "prompt": "生成一张电商商品图，主体清晰，背景干净",
    "model": "batch/gemini-3-pro-image-preview",
    "aspect_ratio": "1:1",
    "resolution": "2K",
    "image_url_list": [
      "https://your-oss.example.com/input.png"
    ]
  }'
```

## 8. Python requests 示例

```python
import requests

service_url = "http://your-image-generation-service/generate"

payload = {
    "out_request_id": "req-demo-001",
    "prompt": "把图片里的裤子改成老欧美风格牛仔裤，保持人物姿态和背景不变",
    "model": "google/gemini-3-pro-image-preview",
    "aspect_ratio": "1:1",
    "resolution": "4K",
    "image_url_list": [
        "https://your-oss.example.com/input.png",
    ],
}

resp = requests.post(service_url, json=payload, timeout=600)
resp.raise_for_status()
data = resp.json()

if data.get("error", {}).get("code"):
    raise RuntimeError(data["error"])

result_url = data["result_image_url"]
image_bytes = requests.get(result_url, timeout=300).content

with open("result.png", "wb") as f:
    f.write(image_bytes)
```

## 9. 如何模拟 `知衣-图生图-combo` 的 batch 行为

Image Generation 服务本身一次请求返回一张结果图。`知衣-图生图-combo` 的 `batch_size` 不是传给服务端的字段，而是在 ComfyUI 节点里循环/并发发送多次请求。

节点任务数：

```text
所有 combo 的有效 prompt 数量之和 * batch_size
```

如果你直接调用服务，也应该这样做：

```python
from concurrent.futures import ThreadPoolExecutor

prompts = [
    "prompt one",
    "prompt two",
]
batch_size = 2
image_url_list = ["https://your-oss.example.com/input.png"]

def call_one(prompt):
    payload = {
        "out_request_id": "req-batch",
        "prompt": prompt,
        "model": "google/gemini-3-pro-image-preview",
        "aspect_ratio": "1:1",
        "resolution": "4K",
        "image_url_list": image_url_list,
    }
    r = requests.post(service_url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["result_image_url"]

tasks = []
for prompt in prompts:
    for _ in range(batch_size):
        tasks.append(prompt)

with ThreadPoolExecutor(max_workers=8) as pool:
    result_urls = list(pool.map(call_one, tasks))
```

## 10. 和 ComfyUI 节点参数的对应关系

| ComfyUI 参数 | 是否发送给服务 | 服务字段 | 说明 |
| --- | --- | --- | --- |
| `model` | 是 | `model` | 会归一化模型名。 |
| `aspect_ratio` | 是 | `aspect_ratio` | 空值会传 `""` 或 `None` 经 client 处理成空字符串。 |
| `image_size` | 是 | `resolution` | 例如 `4K`、`2K`。 |
| `out_request_id` | 是 | `out_request_id` | 空值会变成 `"default"`。 |
| `combo.images` | 是 | `image_url_list` | 先上传 OSS，再传 URL。 |
| `combo.prompts` | 是 | `prompt` | 每条 prompt 单独发请求。 |
| `system_prompt` | 间接发送 | `prompt` | 和 prompt 拼成一个字符串。 |
| `batch_size` | 否 | 无 | 节点本地展开成多次请求。 |
| `max_concurrency` | 否 | 无 | 节点本地并发控制。 |
| `seed_mode` | 否 | 无 | 当前主路径不会把 seed 发给 `FD_GEMINI_URL`。 |
| `seed` | 否 | 无 | 当前只用于节点本地任务 seed 计算和日志，不进服务请求体。 |

## 11. 直接调用时的检查清单

调用前确认：

```text
1. 你知道实际的 FD_GEMINI_URL。
2. image_url_list 里的 URL 图片生成服务能访问。
3. prompt 非空。
4. model 使用最终模型名，例如 google/gemini-3-pro-image-preview。
5. 如果要多张结果图，就多次请求，不要期望一个 n/batch_size 字段。
6. 不要用 gemini-3-pro-image-preview-siphonlab 调这个接口，它走 LiteLLM chat/completions。
```

最推荐的直接调用 payload：

```json
{
  "out_request_id": "业务请求ID",
  "prompt": "最终提示词",
  "model": "google/gemini-3-pro-image-preview",
  "aspect_ratio": "1:1",
  "resolution": "4K",
  "image_url_list": ["https://可访问的输入图URL.png"]
}
```
