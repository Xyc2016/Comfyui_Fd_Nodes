# 图片生成节点调用说明

本文档说明 ComfyUI 自定义节点中“生成图片 / 图生图”相关节点的使用方式，以及这些节点背后的后端调用链路。

文档绝对路径：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/image-generation-node-usage.md
```

## 1. 推荐使用方式

日常业务优先使用下面两类节点：

```text
知衣/图生图:
  知衣-图生图
  知衣-图生图-combo
  知衣-AiStudio图生图-combo

GPT Image:
  FD GPT Multi Image
  FD GPT Image Combo
```

如果只是“1 组图片 + 1 条 prompt 生成多张图”，使用：

```text
知衣-图生图
```

如果有多组图片和多条 prompt，需要批量并发生成，使用：

```text
fd-输入参数组合 -> 知衣-图生图-combo
```

如果要走 AiStudio publish 渠道，使用：

```text
fd-输入参数组合 -> 知衣-AiStudio图生图-combo
```

如果明确要走 GPT Image edits 接口，使用：

```text
FD GPT Multi Image
```

或批量组合：

```text
fd-输入参数组合 -> FD GPT Image Combo
```

## 2. 关键文件绝对路径

节点注册入口：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/nodes.py
```

知衣图生图节点：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/zhiyi_image_to_image_node.py
```

知衣图生图 combo 节点：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/zhiyi_image_to_image_combo_node.py
```

图片组合节点：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/zhiyi_image_combo_node.py
```

AiStudio 图生图 combo 节点：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/aistudio_image_combo_node.py
```

Gemini 图片服务封装：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/gemini_service.py
```

LiteLLM Gemini 图片调用：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/litellm_gemini_image.py
```

GPT Image 节点：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/gpt_image_node.py
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/gpt_multi_image_node.py
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/gpt_image_combo_node.py
```

GPT Image 请求封装：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/gpt_image_request.py
```

旧 Gemini 节点：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/old_gemini_api_node.py
```

配置文件：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/config.py
```

## 3. 节点选择

### 3.1 知衣-图生图

节点类：

```text
ZhiYiImageToImageNode
```

显示名：

```text
知衣-图生图
```

分类：

```text
知衣/图生图
```

适合场景：

```text
最多 6 张输入图 + 1 条 prompt 或 prompt_list，生成一批图片。
```

主要输入：

```text
image_1              必填，主输入图
prompt               必填，默认提示词
model                必填，模型/渠道
aspect_ratio         必填，输出比例
image_size           必填，输出尺寸
batch_size           必填，每条 prompt 生成几张
seed_mode            必填，随机种子 / 固定种子
seed                 必填，固定种子模式下使用
node_switch          可选，1 表示跳过节点
out_request_id       可选，透传给后端的请求 ID
prompt_list          可选，LIST，每条 prompt 都会生成 batch_size 张
image_2 - image_6    可选，额外参考图
system_prompt        可选，会拼到 prompt 前面
```

输出：

```text
image  IMAGE，成功图片会 cat 成 batch 输出
seed   INT，本次使用的基础 seed
```

调用链路：

```text
ComfyUI IMAGE tensor
  -> 选择模型渠道
  -> 如果是 LiteLLM Gemini 渠道：图片转 base64，直接请求 LiteLLM chat/completions
  -> 否则：图片上传 OSS，拿到 image_url_list
  -> 调用 Gemini 图片服务
  -> 下载 result_image_url
  -> 转成 ComfyUI IMAGE tensor
```

### 3.2 fd-输入参数组合

节点类：

```text
ZhiYiImageComboNode
```

显示名：

```text
fd-输入参数组合
```

分类：

```text
知衣/工具
```

作用：

```text
把最多 6 张图片和一组 prompt 打包成 ZHIYI_COMBO，供 combo 生成节点批量使用。
```

主要输入：

```text
image_1              必填
prompt               必填
image_2 - image_6    可选
prompt_list          可选，LIST
```

输出结构：

```json
{
  "images": ["ComfyUI IMAGE tensor 列表"],
  "prompt": "原始 prompt",
  "prompts": ["有效 prompt 列表"]
}
```

### 3.3 知衣-图生图-combo

节点类：

```text
ZhiYiImageToImageComboNode
```

显示名：

```text
知衣-图生图-combo
```

分类：

```text
知衣/图生图
```

适合场景：

```text
多个 ZHIYI_COMBO 并发生成，每个 combo 可以有多张图和多条 prompt。
```

主要输入：

```text
model                必填
aspect_ratio         必填
image_size           必填
batch_size           必填，每条 prompt 生成几张
max_concurrency      必填，同时请求数上限
seed_mode            必填
seed                 必填
out_request_id       可选
combo_1 - combo_8    可选，至少连接一个
system_prompt        可选
```

输出：

```text
image  IMAGE list，OUTPUT_IS_LIST=True，每个成功请求输出一张
seed   INT
log    STRING，包含成功/失败统计和后端 URL
```

调用链路：

```text
ZHIYI_COMBO
  -> 展开 combo.images 中的 batch 图片
  -> 对 combo.prompts 逐条生成任务
  -> 每条 prompt 乘以 batch_size
  -> 根据 model 判断走 LiteLLM 还是 Gemini 图片服务
  -> 并发请求
  -> 输出成功图片列表和日志
```

### 3.4 知衣-AiStudio图生图-combo

节点类：

```text
ZhiYiAiStudioImageComboNode
```

显示名：

```text
知衣-AiStudio图生图-combo
```

分类：

```text
知衣/图生图
```

适合场景：

```text
需要走内部 AiStudio publish 测试渠道生成图片。
```

调用链路：

```text
ZHIYI_COMBO
  -> 展开图片
  -> 图片上传 OSS
  -> 组装 publish payload
  -> POST FD_AISTUDIO_PUBLISH_URL
  -> 从响应 data.url 读取结果图 URL
  -> 下载结果图
  -> 输出 IMAGE list、seed、log
```

请求体形状：

```json
{
  "type": "AiStudio",
  "payload": {
    "prompt": "system_prompt 和 prompt 拼接后的文本",
    "image": ["https://oss/path/input.png"],
    "image_size": "4K",
    "aspect_ratio": "1:1"
  },
  "timeout": 300000
}
```

注意：

```text
model、seed、out_request_id 目前只是为了兼容旧 UI，AiStudio publish 接口当前不传这些字段。
```

更详细说明见：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/aistudio-node-usage.md
```

### 3.5 FD GPT Multi Image

节点类：

```text
FD_GPTMultiImage
```

显示名：

```text
FD GPT Multi Image
```

分类：

```text
GPT Image
```

适合场景：

```text
最多 6 张图片，直接调用 GPT Image edits API 进行图生图。
```

调用链路：

```text
ComfyUI IMAGE tensor
  -> 图片压缩到不超过 4096 * 4096 总像素
  -> multipart/form-data 上传到 {FD_LITELLM_BASE_URL}/v1/images/edits
  -> 解析返回的 url 或 b64_json
  -> 转成 ComfyUI IMAGE
```

请求地址：

```text
{FD_LITELLM_BASE_URL}/v1/images/edits
```

请求方式：

```text
POST multipart/form-data
Authorization: Bearer {FD_LITELLM_API_KEY}
```

表单字段：

```text
model       gpt-image-2
prompt      prompt 文本
size        根据 image_size + aspect_ratio 映射出的尺寸，例如 2048x2048
user        out_request_id
quality     medium
image       多个 multipart 文件字段
```

### 3.6 FD GPT Image Combo

节点类：

```text
FD_GPTImageComboNode
```

显示名：

```text
FD GPT Image Combo
```

分类：

```text
GPT Image
```

适合场景：

```text
用 ZHIYI_COMBO 批量调用 GPT Image edits API。
```

调用链路和 `FD GPT Multi Image` 一致，只是输入来自 `fd-输入参数组合`，并支持：

```text
combo_1 - combo_8
batch_size
max_concurrency
system_prompt
```

输出：

```text
image  IMAGE list
seed   INT
log    STRING
```

### 3.7 FD GTP Image

节点类：

```text
FD_GTPImage
```

显示名：

```text
FD GTP Image
```

分类：

```text
image/generation
```

这是底层 GPT Image edits API 节点，适合简单直接调用，不支持 combo。

主要输入：

```text
out_request_id
prompt
model=gpt-image-2
resolution=1K/2K/4K
images
aspect_ratio
```

输出：

```text
IMAGE
STRING  revised_prompt 或 message
STRING  result_url
```

### 3.8 FD Gemini Image

节点类：

```text
FD_GeminiImage
```

显示名：

```text
FD Gemini Image
```

分类：

```text
image/generation
```

这是旧 Gemini 图片节点，直接调用 `FD_GEMINI_URL`。

调用链路：

```text
ComfyUI IMAGE tensor
  -> 图片压缩到不超过 3072 * 3072 总像素
  -> 上传到 OSS
  -> POST FD_GEMINI_URL
  -> 后端返回 result_image_url
  -> 下载结果图
```

请求体形状：

```json
{
  "out_request_id": "default",
  "prompt": "生成提示词",
  "model": "google/gemini-3-pro-image-preview",
  "aspect_ratio": "1:1",
  "resolution": "2K",
  "image_url_list": ["https://oss/path/input.png"]
}
```

返回体期望：

```json
{
  "result_image_url": "https://oss/path/output.png",
  "message": "文本消息",
  "cost_time": 12.3
}
```

## 4. 模型和渠道

知衣图生图节点支持的常见 model：

```text
google/gemini-3-pro-image-preview
google/gemini-3.1-flash-image-preview
google/gemini-2.5-flash-image-preview
google/gemini-3-pro-image-preview-official
gemini-3-pro-image-preview
batch/gemini-3-pro-image-preview
gemini-3.1-flash-image-preview
gemini-2.5-flash-image-preview
gemini-3-pro-image-preview-official
gemini-3-pro-image-preview-aistudio
gemini-3-pro-image-preview-siphonlab
gemini-nano-banana-2.1
```

模型名归一化规则在：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/gemini_service.py
```

部分模型会走 LiteLLM Gemini 图片调用，判断逻辑在：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/litellm_gemini_image.py
```

当前 LiteLLM Gemini 渠道模型包括：

```text
gemini-3-pro-image-preview-siphonlab
```

新模型 `gemini-nano-banana-2.1` 走非 LiteLLM 的 Gemini image-server。它支持 `1K`、`2K`、`4K`；节点中的 `720P` 会映射为 `1K`，`1080P` 会映射为 `2K`。编辑场景默认开启色彩偏移修正，关闭时会显式传 `enable_color_bias_correction: false`。

这类 LiteLLM 模型不上传 OSS 给 Gemini 图片服务，而是把图片转 base64，组装成多模态 messages 后请求：

```text
{FD_LITELLM_BASE_URL}/v1/chat/completions
```

非 LiteLLM Gemini 渠道会走：

```text
FD_GEMINI_URL
```

并使用 `image_url_list` 传 OSS 图片 URL。

## 5. 环境变量

### 5.1 Gemini 图片服务

```text
FD_GEMINI_URL
```

用途：

```text
旧 Gemini 节点、知衣图生图非 LiteLLM Gemini 渠道的服务地址。
```

### 5.2 LiteLLM / OpenAI 兼容服务

```text
FD_LITELLM_BASE_URL
FD_LITELLM_API_KEY
```

用途：

```text
GPT Image edits API
Seedream images/generations API
LiteLLM Gemini chat/completions 图片生成渠道
```

### 5.3 OSS 上传

```text
FD_OSS_ACCESS_KEY_ID
FD_OSS_ACCESS_KEY_SECRET
FD_OSS_BUCKET_NAME
FD_OSS_ENDPOINT
FD_OSS_URL_PREFIX
```

用途：

```text
Gemini 图片服务、AiStudio、Flux2Klein、Z-Image-Turbo、Seedream 需要先把输入图片上传 OSS。
```

不同节点使用的 OSS 路径前缀：

```text
FD_OSS_URL_PATH_PREFIX_GEMINI       默认 devops/comfyui/segment_img
FD_OSS_URL_PATH_PREFIX_BEFORE_GEN   默认 devops/comfyui/segment_img
FD_OSS_URL_PATH_PREFIX_FLUX         默认 devops/comfyui/segment_img
```

### 5.4 AiStudio publish

```text
FD_AISTUDIO_PUBLISH_URL
```

默认值：

```text
http://121.40.67.98:2003/api/tasks/publish
```

### 5.5 Flux2Klein

```text
FD_FLUX2KLEIN_URL
FD_FLUX2KLEIN_USERNAME
FD_FLUX2KLEIN_PASSWORD
```

### 5.6 Z-Image-Turbo

```text
FD_Z_IMAGE_TURBO_URL
FD_Z_IMAGE_TURBO_USERNAME
FD_Z_IMAGE_TURBO_PASSWORD
```

### 5.7 Webhook

```text
FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL
FD_GEMINI_WEBHOOK_URL
```

用途：

```text
发送请求摘要、完整响应摘要等通知。未配置时不会发送。
```

## 6. 后端调用方式汇总

### 6.1 知衣图生图非 LiteLLM Gemini 渠道

请求地址：

```text
FD_GEMINI_URL
```

请求方式：

```text
POST application/json
```

请求体：

```json
{
  "out_request_id": "req-001",
  "prompt": "生成提示词",
  "model": "google/gemini-3-pro-image-preview",
  "aspect_ratio": "1:1",
  "resolution": "4K",
  "image_url_list": [
    "https://oss/path/input-1.png",
    "https://oss/path/input-2.png"
  ]
}
```

响应体：

```json
{
  "result_image_url": "https://oss/path/result.png",
  "message": "生成说明",
  "cost_time": 20.5
}
```

节点随后会下载 `result_image_url`，转成 ComfyUI `IMAGE` 输出。

### 6.2 LiteLLM Gemini 图片渠道

请求地址：

```text
{FD_LITELLM_BASE_URL}/v1/chat/completions
```

请求方式：

```text
POST application/json
Authorization: Bearer {FD_LITELLM_API_KEY}
```

请求特点：

```text
输入图不会先传 OSS，而是转 base64 放进 messages。
model 使用节点选择的 aistudio / siphonlab 渠道模型。
aspect_ratio、image_size、seed、out_request_id 会一起传入调用封装。
```

实现文件：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/litellm_gemini_image.py
```

### 6.3 GPT Image edits

请求地址：

```text
{FD_LITELLM_BASE_URL}/v1/images/edits
```

请求方式：

```text
POST multipart/form-data
Authorization: Bearer {FD_LITELLM_API_KEY}
```

表单字段：

```text
model       gpt-image-2
prompt      生成提示词
size        1024x1024 / 2048x2048 / 2880x2880 等
quality     medium
user        out_request_id，可选
image       一个或多个 PNG 文件
```

响应支持两种图片形式：

```text
data[0].url
data[0].b64_json
```

节点优先下载 `data[0].url`；如果没有 URL，则解码 `b64_json`。

超时/失败策略：

```text
主模型 gpt-image-2 最多尝试 3 次。
如果发生 TIMEOUT，或主模型多次失败，会 fallback 到 gpt-image-2-azure。
fallback 请求会额外带 n 字段。
```

### 6.4 AiStudio publish

请求地址：

```text
FD_AISTUDIO_PUBLISH_URL
```

请求方式：

```text
POST application/json
```

请求体：

```json
{
  "type": "AiStudio",
  "payload": {
    "prompt": "生成提示词",
    "image": ["https://oss/path/input.png"],
    "image_size": "4K",
    "aspect_ratio": "1:1"
  },
  "timeout": 300000
}
```

响应体要求：

```json
{
  "success": true,
  "taskId": "task-id",
  "data": {
    "url": "https://oss/path/result.png"
  }
}
```

### 6.5 Flux2Klein

节点类：

```text
FD_Flux2KleinGenImage
```

显示名：

```text
FD Flux2Klein Gen Image
```

请求地址：

```text
service_url 输入项，默认 FD_FLUX2KLEIN_URL
```

认证：

```text
HTTP Basic Auth:
FD_FLUX2KLEIN_USERNAME
FD_FLUX2KLEIN_PASSWORD
```

请求体：

```json
{
  "out_request_id": "req-001",
  "prompt": "生成提示词",
  "seed": 0,
  "ratio": "1:1",
  "size": "2K",
  "images": ["https://oss/path/input.png"]
}
```

响应体要求：

```json
{
  "urls": ["https://oss/path/result.png"],
  "status": "success",
  "cost_time": 45.2
}
```

当前节点只取 `urls[0]`。

### 6.6 Z-Image-Turbo

节点类：

```text
FD_ZImageTurboGenImage
```

显示名：

```text
FD Z-Image-Turbo Gen Image
```

请求地址：

```text
FD_Z_IMAGE_TURBO_URL
```

认证：

```text
HTTP Basic Auth:
FD_Z_IMAGE_TURBO_USERNAME
FD_Z_IMAGE_TURBO_PASSWORD
```

请求体：

```json
{
  "out_request_id": "req-001",
  "prompt": "生成提示词",
  "seed": 42,
  "ratio": "1:1",
  "size": "1K",
  "strength": 0.25,
  "guidance_scale": 0.0,
  "num_inference_steps": 9,
  "num_images": 1,
  "images": ["https://oss/path/input.png"]
}
```

响应体要求：

```json
{
  "urls": ["https://oss/path/result.png"]
}
```

输出：

```text
IMAGE
STRING result_url
```

### 6.7 Seedream

节点类：

```text
FD_SeedreamImage
```

显示名：

```text
FD Seedream Image
```

默认调用 image-server 统一接口（`FD_SEEDREAM_BACKEND=image_generation`，默认值），不再直连 LiteLLM：

- 文生图：`POST {FD_IMAGE_GENERATE_URL}`（默认 `http://image-generation.online-server-gray.svc.cluster.local/image/generate`，可从 `FD_GEMINI_URL` / `FD_GPT_IMAGE_EDIT_URL` 推导）
- 图生图：输入图降采样（≤4096×4096 像素）上传 OSS 后 `POST {FD_GPT_IMAGE_EDIT_URL}`（默认 .../image/edit），channel 区分模型

请求体（文生图）：

```json
{
  "channel": "doubao-seedream-5.0-pro",
  "prompt": "生成提示词",
  "size": "4K",
  "ratio": "3:4",
  "resize": true
}
```

请求体（图生图）：

```json
{
  "channel": "doubao-seedream-5.0-lite",
  "prompt": "编辑提示词",
  "size": "2K",
  "ratio": "9:21",
  "resize": true,
  "image_url_list": ["https://oss/path/input.png"]
}
```

响应体要求：

```json
{
  "status": true,
  "result_image_url": "https://oss/path/result.png"
}
```

参数说明：

- `size`：`1K`/`2K`/`3K`/`4K`，默认 `2K`。image-server 仅支持 1K/2K/4K，节点将 `3K` 映射为 `2K` 请求并记录 warning（豆包 Pro/Lite 均无 3K 档）
- `ratio`：`1:1`、`2:3`、`3:2`、`3:4`、`4:3`、`4:5`、`5:4`、`16:9`、`9:16`、`21:9`、`9:21`
- `resize`：固定 `true`。Pro + 4K 由 image-server 按同一比例请求上游 2K 并超分到长边 4096（解决 Pro 不支持 3K/4K 的 400 问题）
- `output_format` 参数保留兼容，image-server 路径不上送
- `model` 参数（`doubao-seedream-5.0-pro` / `doubao-seedream-5.0-lite` / `doubao-seedream-5.0-flash`）作为 `channel` 透传；显式 channel 失败不跨模型回退。flash 参数与分辨率语义与 pro 完全一致，仅速度更快

回退开关：`FD_SEEDREAM_BACKEND=litellm` 时走旧直连 `{FD_LITELLM_BASE_URL}/v1/images/generations`（像素换算、行为同旧版，仅紧急回退用）。

## 7. 图片处理规则

### 7.1 输入图片转 OSS URL

大部分非 GPT Image edits 节点会执行：

```text
ComfyUI IMAGE tensor
  -> numpy uint8
  -> PNG bytes
  -> md5 命名
  -> 上传 OSS
  -> 得到公网/内网可访问 URL
```

Gemini 图片服务默认路径：

```text
{FD_OSS_URL_PATH_PREFIX_GEMINI}/{md5}.png
```

AiStudio / Seedream 默认路径：

```text
{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{md5}.png
```

Flux2Klein / Z-Image-Turbo 默认路径：

```text
{FD_OSS_URL_PATH_PREFIX_FLUX}/{md5}.png
```

### 7.2 输入图片降采样

不同节点为了控制请求大小，会做不同程度的降采样：

```text
FD_GeminiImage:         总像素不超过 3072 * 3072
ZhiYiImageToImageNode:  Gemini service client 内部总像素不超过 3072 * 3072
FD_GTPImage:            总像素不超过 4096 * 4096
FD_GPTMultiImage:       总像素不超过 4096 * 4096
FD_GPTImageComboNode:   总像素不超过 4096 * 4096
FD_Flux2KleinGenImage:  总像素不超过 2048 * 2048
FD_SeedreamImage:       图生图总像素不超过 4096 * 4096（image-server 路径，seedream client 内降采样）
```

### 7.3 输出图片

所有图片生成节点最终都会：

```text
下载后端返回的结果图 URL
  -> PIL Image RGB
  -> numpy float32 / 255.0
  -> torch.Tensor
  -> ComfyUI IMAGE
```

## 8. 尺寸和比例

### 8.1 Gemini / 知衣图生图

`image_size` 常用选项：

```text
4K
2K
1080P
720P
```

`aspect_ratio` 常用选项：

```text
""
1:1
2:3
3:2
3:4
4:3
4:5
5:4
16:9
9:16
21:9
```

combo 节点额外支持：

```text
1:4
1:8
4:1
8:1
```

### 8.2 GPT Image

GPT Image 节点会把 `image_size + aspect_ratio` 映射为具体像素尺寸。

映射函数：

```text
/Users/chenjunhong/Documents/web_project/Comfyui_Fd_Nodes/src/Comfyui_Fd_Nodes/utils/gpt_image_size.py
```

示例：

```text
1K + 1:1   -> 1024x1024
2K + 1:1   -> 2048x2048
4K + 1:1   -> 2880x2880
1K + 3:4   -> 768x1024
2K + 3:4   -> 1536x2048
4K + 3:4   -> 2160x2880
1K + 9:16  -> 720x1280
2K + 9:16  -> 1152x2048
4K + 9:16  -> 2160x3840
```

## 9. 并发和 batch 规则

### 9.1 单节点 batch

`知衣-图生图` 和 `FD GPT Multi Image` 的任务数：

```text
有效 prompt 数量 * batch_size
```

如果没有 `prompt_list`，有效 prompt 数量为 1。

### 9.2 combo batch

`知衣-图生图-combo`、`知衣-AiStudio图生图-combo`、`FD GPT Image Combo` 的任务数：

```text
所有 combo 的有效 prompt 数量之和 * batch_size
```

`max_concurrency` 控制同时请求数上限。

### 9.3 seed 规则

随机种子模式：

```text
每个请求随机生成 seed。
```

固定种子模式：

```text
以输入 seed 为基础，按任务顺序递增。
```

注意：

```text
目前部分后端不一定真正使用 seed。seed 字段更多是为了日志追踪和兼容 UI。
```

## 10. 常见接线示例

### 10.1 单组图生图

```text
Load Image
  -> 知衣-图生图.image_1

设置:
  prompt       = 生成提示词
  model        = google/gemini-3-pro-image-preview
  image_size   = 2K 或 4K
  aspect_ratio = 1:1 / 3:4 / 9:16
  batch_size   = 1
```

### 10.2 多组图片批量生成

```text
Load Image A + prompt A
  -> fd-输入参数组合.combo
  -> 知衣-图生图-combo.combo_1

Load Image B + prompt B
  -> fd-输入参数组合.combo
  -> 知衣-图生图-combo.combo_2

知衣-图生图-combo:
  batch_size      = 1 或更多
  max_concurrency = 8 / 16
```

### 10.3 AiStudio 批量生成

```text
Load Image
  -> fd-输入参数组合
  -> 知衣-AiStudio图生图-combo
```

要求：

```text
FD_AISTUDIO_PUBLISH_URL 已配置或使用默认值。
OSS 参数完整。
```

### 10.4 GPT Image 图生图

```text
Load Image
  -> FD GPT Multi Image.image_1

设置:
  model        = gpt-image-2
  image_size   = 2K / 4K
  aspect_ratio = 1:1 / 3:4 / 9:16
```

要求：

```text
FD_LITELLM_BASE_URL
FD_LITELLM_API_KEY
```

## 11. 常见错误

### 11.1 未配置 OSS

错误示例：

```text
未配置 OSS 上传参数: FD_OSS_ACCESS_KEY_ID, ...
```

原因：

```text
节点需要先把输入图上传 OSS，但 OSS 环境变量不完整。
```

影响节点：

```text
知衣-图生图
知衣-图生图-combo
知衣-AiStudio图生图-combo
FD Gemini Image
FD Flux2Klein Gen Image
FD Z-Image-Turbo Gen Image
FD Seedream Image
```

GPT Image edits 节点不需要 OSS 上传输入图，因为它走 multipart/form-data。

### 11.2 未配置 LiteLLM

错误示例：

```text
未配置 base_url，请在环境变量 FD_LITELLM_BASE_URL 中设置
未配置 api_key，请在环境变量 FD_LITELLM_API_KEY 中设置
```

影响节点：

```text
FD GTP Image
FD GPT Multi Image
FD GPT Image Combo
FD Seedream Image
LiteLLM Gemini 图片渠道
```

### 11.3 没有输入图片

错误示例：

```text
未提供图片
未提供任何组合输入，请连接至少一个 combo
所有组合均无有效图片
```

处理方式：

```text
确认 image_1 或 combo_1 已连接。
确认 combo 内至少有一张有效 IMAGE。
```

### 11.4 后端没有返回图片 URL

错误示例：

```text
响应缺少 data.url
响应缺少 result_image_url
GPT Image API returned no usable image payload
```

处理方式：

```text
检查后端响应格式是否符合对应节点的期望。
检查生成服务是否成功完成任务。
```

### 11.5 TIMEOUT

原因：

```text
后端生成耗时过长、网络超时、模型服务繁忙。
```

GPT Image edits 节点会尝试 fallback：

```text
gpt-image-2 -> gpt-image-2-azure
```

其他节点一般会直接抛出归一化后的错误。

## 12. 给调用方的最小说明

如果只需要告诉别人怎么用，给这段即可：

```text
1. 单组图生图用“知衣-图生图”。
2. 多组批量用“fd-输入参数组合”打包图片和 prompt，再接“知衣-图生图-combo”。
3. AiStudio 渠道用“fd-输入参数组合”接“知衣-AiStudio图生图-combo”。
4. GPT Image 渠道用“FD GPT Multi Image”；批量用“fd-输入参数组合”接“FD GPT Image Combo”。
5. Gemini 普通渠道会先上传图片到 OSS，然后 POST FD_GEMINI_URL。
6. GPT Image 渠道直接 multipart POST {FD_LITELLM_BASE_URL}/v1/images/edits。
7. AiStudio 渠道会上传 OSS 后 POST FD_AISTUDIO_PUBLISH_URL，读取响应 data.url。
8. 必须配置对应环境变量：OSS、FD_GEMINI_URL、FD_LITELLM_BASE_URL、FD_LITELLM_API_KEY、FD_AISTUDIO_PUBLISH_URL 等。
```
