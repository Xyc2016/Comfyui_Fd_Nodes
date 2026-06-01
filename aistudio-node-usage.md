# 知衣 AiStudio 图生图 Combo 节点说明

本文档说明 ComfyUI 自定义节点 `知衣-AiStudio图生图-combo` 背后的调用方式、输入输出、环境变量和接口数据格式。

代码绝对路径：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/aistudio_image_combo_node.py
```

节点类名：

```text
ZhiYiAiStudioImageComboNode
```

节点显示名：

```text
知衣-AiStudio图生图-combo
```

## 1. 总体调用链路

这个节点不是直接调用 LiteLLM、OpenAI 兼容接口或 Gemini 接口。它走的是内部 AiStudio publish 任务接口。

整体链路如下：

```text
ComfyUI combo 输入
  -> 读取 combo.images / combo.prompts
  -> 将图片 tensor 转为 PNG bytes
  -> 上传图片到 OSS
  -> 得到 OSS 图片 URL 列表
  -> 拼接 system_prompt 和 prompt
  -> 调用 AiStudio publish 接口
  -> 从响应中读取 data.url
  -> 下载 data.url 对应的生成结果图
  -> 转为 ComfyUI IMAGE 输出
```

可以理解为：

```text
输入图 + prompt -> OSS 图片 URL + prompt -> AiStudio publish -> 结果图 URL -> 下载结果图
```

## 2. 服务地址

AiStudio publish 接口地址来自环境变量：

```text
FD_AISTUDIO_PUBLISH_URL
```

默认值：

```text
http://121.40.67.98:2003/api/tasks/publish
```

配置代码绝对路径：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/config.py
```

如果 `FD_AISTUDIO_PUBLISH_URL` 为空，节点会报错：

```text
未配置 AiStudio publish URL，请设置环境变量 FD_AISTUDIO_PUBLISH_URL
```

## 3. OSS 依赖

节点会先上传输入图片到 OSS，再把 OSS URL 发给 AiStudio publish 接口。因此运行环境必须配置 OSS 参数。

必需环境变量：

```text
FD_OSS_ACCESS_KEY_ID
FD_OSS_ACCESS_KEY_SECRET
FD_OSS_BUCKET_NAME
FD_OSS_ENDPOINT
FD_OSS_URL_PREFIX
```

图片上传路径前缀：

```text
FD_OSS_URL_PATH_PREFIX_BEFORE_GEN
```

默认值：

```text
devops/comfyui/segment_img
```

上传后的文件路径格式：

```text
{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{图片bytes的md5}.png
```

最终传给 AiStudio 的图片地址格式：

```text
{FD_OSS_URL_PREFIX}{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{md5}.png
```

## 4. 节点输入

### 必填输入

```text
model
aspect_ratio
image_size
batch_size
max_concurrency
seed_mode
seed
```

当前 `model` 只有一个选项：

```text
nano-banana-pro
```

可选比例：

```text
"", "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3",
"4:5", "5:4", "8:1", "9:16", "16:9", "21:9"
```

可选尺寸：

```text
4K, 2K, 1080P, 720P
```

### 可选输入

```text
out_request_id
combo_1
combo_2
combo_3
combo_4
combo_5
combo_6
combo_7
combo_8
system_prompt
```

每个 `combo_x` 是一个 `ZHIYI_COMBO`，节点会读取其中的：

```text
images
prompts
prompt
```

读取规则：

```python
combo_images = combo.get("images", [])
combo_prompts = combo.get("prompts") or [combo.get("prompt", "")]
```

如果 `prompts` 存在，则使用 `prompts` 列表；否则使用单个 `prompt`。

## 5. Prompt 拼接方式

如果同时有 `system_prompt` 和普通 prompt，最终 prompt 会拼成：

```text
{system_prompt}

{prompt}
```

如果只有其中一个，则直接使用非空的那个。

示例：

```text
system_prompt = "你是专业电商图片生成助手"
prompt = "给模特换上白色衬衫"
```

最终传给接口：

```text
你是专业电商图片生成助手

给模特换上白色衬衫
```

## 6. 请求体格式

节点实际调用的是：

```http
POST {FD_AISTUDIO_PUBLISH_URL}
Content-Type: application/json
```

请求超时时间：

```text
600 秒
```

请求体结构：

```json
{
  "type": "AiStudio",
  "payload": {
    "prompt": "最终拼好的提示词",
    "image": [
      "https://example.com/input-1.png",
      "https://example.com/input-2.png"
    ],
    "image_size": "4K",
    "aspect_ratio": "3:4"
  },
  "timeout": 300000
}
```

说明：

- `payload.prompt` 是最终 prompt。
- `payload.image` 是 OSS 图片 URL 数组。
- `payload.image_size` 是节点选择的图片尺寸。
- `payload.aspect_ratio` 只有在节点选择了非空比例时才会传。
- 顶层 `timeout` 固定为 `300000`。

当 `aspect_ratio` 为空字符串时，请求体不会包含 `aspect_ratio` 字段：

```json
{
  "type": "AiStudio",
  "payload": {
    "prompt": "最终拼好的提示词",
    "image": [
      "https://example.com/input.png"
    ],
    "image_size": "4K"
  },
  "timeout": 300000
}
```

## 7. model / seed / out_request_id 的注意点

当前节点 UI 中有：

```text
model
seed_mode
seed
out_request_id
```

但这些字段目前不会传给 AiStudio publish 接口。

也就是说，实际请求体里不会出现：

```text
model
seed
out_request_id
```

它们当前的作用：

- `model`：保留 UI 兼容和日志展示，目前固定为 `nano-banana-pro`。
- `seed_mode` / `seed`：节点内部会计算并输出 seed，但不会真正控制 AiStudio 生成结果。
- `out_request_id`：兼容旧节点输入，当前不传给接口。

固定种子模式下，节点会按任务序号递增 seed：

```text
task_seed = actual_seed + task_idx
```

但这个 `task_seed` 只进入日志，不进入 publish 请求体。

## 8. 响应格式

节点期望 AiStudio publish 接口返回：

```json
{
  "taskId": "task-1",
  "success": true,
  "data": {
    "url": "https://example.com/result.png"
  }
}
```

节点处理逻辑：

1. 检查 `success` 必须为 `true`。
2. 检查 `data` 必须是 object。
3. 检查 `data.url` 必须存在。
4. 使用 `requests.get(data.url, timeout=300)` 下载结果图。
5. 将结果图转为 ComfyUI `IMAGE` tensor。

如果 `success` 不是 `true`，会报错：

```text
API 返回失败: ...
```

如果缺少 `data.url`，会报错：

```text
响应缺少 data.url
```

## 9. Batch 和并发规则

节点会把任务展开为多个 publish 请求。

展开规则：

```text
每个 combo
  -> 上传该 combo 内所有图片，得到 image_urls
  -> 每个 prompt
    -> 重复 batch_size 次
      -> 创建一个 publish 请求
```

任务数计算：

```text
任务总数 = combo 数量 × 每个 combo 的 prompt 数量 × batch_size
```

示例：

```text
combo_1 有 2 个 prompts
combo_2 有 3 个 prompts
batch_size = 4
```

总请求数：

```text
(2 + 3) × 4 = 20
```

这些请求通过 `ThreadPoolExecutor` 并发发送，并发上限由：

```text
max_concurrency
```

控制。

## 10. 节点输出

节点返回 3 个输出：

```text
image
seed
log
```

定义：

```text
RETURN_TYPES = ("IMAGE", "INT", "STRING")
RETURN_NAMES = ("image", "seed", "log")
OUTPUT_IS_LIST = (True, False, False)
```

说明：

- `image`：成功生成的图片列表。
- `seed`：本次节点计算出来的 actual_seed。
- `log`：每个请求的成功/失败摘要，包括 taskId 和 result url。

因为 `OUTPUT_IS_LIST = (True, False, False)`，第一个输出是 IMAGE list，不是普通单个 batch tensor。

## 11. 日志内容

每次请求会打印一个日志摘要，包含：

```text
type
model
timeout
prompt
image_count
aspect_ratio
image_size
seed
out_request_id
note
```

其中 `note` 会提示：

```text
model、seed、out_request_id 仅兼容旧节点 UI，AiStudio publish 接口当前不传这些字段
```

最终 `log` 输出会包含类似：

```text
总计: 2/2 成功, url=http://121.40.67.98:2003/api/tasks/publish
[请求 1] 成功 (taskId=task-1, url=https://example.com/result-1.png)
[请求 2] 成功 (taskId=task-2, url=https://example.com/result-2.png)
```

## 12. 常见失败原因

### 1. 未配置 OSS

报错类似：

```text
未配置 OSS 上传参数: FD_OSS_ACCESS_KEY_ID, FD_OSS_ACCESS_KEY_SECRET, ...
```

原因：

节点必须先上传输入图到 OSS，缺少 OSS 配置就无法继续。

### 2. 未连接 combo

报错：

```text
未提供任何组合输入，请连接至少一个 combo
```

原因：

至少需要连接一个 `combo_1` 到 `combo_8`。

### 3. combo 中没有图片

报错类似：

```text
[combo_1] 预处理失败，跳过: RuntimeError: UNKNOWN: 无图片
```

原因：

combo 里没有 `images`，或者图片为空。

### 4. prompt 为空

报错：

```text
prompt 不能为空
```

原因：

最终拼接后的 prompt 是空字符串。

### 5. publish 接口失败

报错类似：

```text
API 请求失败: 500
...
```

或：

```text
API 返回失败: ...
```

原因：

AiStudio publish 服务返回 HTTP 错误，或 JSON 里 `success` 不是 `true`。

### 6. 结果图下载失败

报错来自：

```text
requests.get(result_url, timeout=300)
```

原因：

`data.url` 无法访问、过期、权限不足，或返回的不是合法图片。

## 13. 最小请求示例

假设输入图已经上传到 OSS，最终发给 AiStudio 的最小请求类似：

```bash
curl -X POST "http://121.40.67.98:2003/api/tasks/publish" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "AiStudio",
    "payload": {
      "prompt": "给模特换上白色衬衫",
      "image": [
        "https://oss.example.com/devops/comfyui/segment_img/input.png"
      ],
      "image_size": "4K"
    },
    "timeout": 300000
  }'
```

带比例的请求：

```bash
curl -X POST "http://121.40.67.98:2003/api/tasks/publish" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "AiStudio",
    "payload": {
      "prompt": "给模特换上白色衬衫",
      "image": [
        "https://oss.example.com/devops/comfyui/segment_img/input.png"
      ],
      "image_size": "4K",
      "aspect_ratio": "3:4"
    },
    "timeout": 300000
  }'
```

## 14. 关键结论

这个节点的核心不是“模型 API 调用”，而是“OSS 图片上传 + 内部 AiStudio publish 任务接口 + 结果 URL 下载”。

实际传给 publish 接口的核心字段只有：

```text
type
payload.prompt
payload.image
payload.image_size
payload.aspect_ratio
timeout
```

当前不会传：

```text
model
seed
out_request_id
```
