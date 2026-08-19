# SceneMind-X Self-hosted 接入｜视觉与自然语言处理课程项目

## 中文版

### 1. 定位与边界

Self-hosted 是高级可选接入方式，适合已经拥有 Linux GPU Server、并希望将远端 VLM 与 Embedding 接入 SceneMind-X 的使用者。视觉与自然语言处理课程评阅不要求服务器，默认推荐 Cloud API。课程包保留 Self-hosted Provider 客户端，但不附带用户历史私人服务器 runtime、模型权重、服务器环境、地址、账号或密钥；远端兼容服务需要使用者自行部署。

### 2. 客户端所需 API 合同

SceneMind-X 通过两个独立 HTTP endpoint 连接远端能力：

- VLM endpoint：`GET /health` 返回 `vlm.status`、`vlm.loaded`、模型和 revision；任务端点包括 `/analyze`、`/vqa`、`/describe`、`/generate`、`/compare` 与 `/course-prompt`，响应使用统一 `result` 对象。
- Embedding endpoint：`GET /health` 返回 `embedding.status`、`embedding.loaded`、模型、revision 与维度；`POST /embed/text`、`/embed/image`、`/embed/multimodal` 返回 2048D 向量及维度信息。

当前客户端按 `Qwen/Qwen3-VL-4B-Instruct` 和 `Qwen/Qwen3-VL-Embedding-2B` 合同组织状态。远端实现可以自行选择，只要严格兼容上述请求与响应；SceneMind-X 不会自动在远端安装或启动模型。

### 3. 安全网络连接

建议远端服务仅监听服务器 `127.0.0.1`，通过 VPN、SSH tunnel 或其他受控私网连接映射到运行 SceneMind-X 的电脑。示例仅使用占位符：

```powershell
ssh -N -L <本地VLM端口>:127.0.0.1:<远端VLM端口> -L <本地Embedding端口>:127.0.0.1:<远端Embedding端口> <服务器别名>
```

建立连接后，在启动 SceneMind-X 的同一 PowerShell 会话中配置：

```powershell
$env:SCENEMINDX_VLM_ENDPOINT='http://127.0.0.1:<本地VLM端口>'
$env:SCENEMINDX_E1_EMBEDDING_ENDPOINT='http://127.0.0.1:<本地Embedding端口>'
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_scenemindx.ps1
```

不要把服务直接开放到公网，不要把私钥、Token、真实地址或账号写入工程。

### 4. 选择与 Ready 检查

1. 先在本机分别访问两个 endpoint 的 `/health`，确认 VLM 与 Embedding 已加载。
2. 启动 SceneMind-X，打开模型接入面板。
3. 选择“服务器映射（课程演示）”这一 Self-hosted Provider 显示项。
4. 点击“重新检查服务器连接”，确认 VLM 和 Embedding 分别为 Ready，Embedding 维度为 2048。
5. 保存接入选择，再用已有图片完成一次 VQA 和一次检索。

VLM 与 Embedding 能力独立显示；其中一个不可用时不会伪装为整体 Ready。Self-hosted 与 Local 共享模型表示合同，但各自维护物理 Faiss，Cloud 2560D 索引不可混用。

### 5. 停止与隔离

停止自己建立的 SSH tunnel 和自己部署的远端服务。共享服务器上只能处理本项目受控进程，不得按名称批量停止其他项目。模型、环境、日志、缓存和索引均由远端服务维护者自行隔离和备份。

---

## English Version

### 1. Positioning and boundary

Self-hosted is advanced and optional for users who already operate a Linux GPU server. It is not required for grading the Visual and Natural Language Processing course project; Cloud API remains recommended. The package keeps the Self-hosted client but includes no private historical runtime, model weights, server environment, address, account, or key. Users deploy their own compatible remote services.

### 2. Required API contract

SceneMind-X connects to two independent HTTP endpoints:

- VLM: `GET /health` reports `vlm.status`, `vlm.loaded`, model, and revision. Task routes include `/analyze`, `/vqa`, `/describe`, `/generate`, `/compare`, and `/course-prompt`, returning a common `result` object.
- Embedding: `GET /health` reports `embedding.status`, `embedding.loaded`, model, revision, and dimension. `POST /embed/text`, `/embed/image`, and `/embed/multimodal` return a 2048D vector and dimension metadata.

The client expects the `Qwen/Qwen3-VL-4B-Instruct` and `Qwen/Qwen3-VL-Embedding-2B` contracts. Any remote implementation is acceptable when request and response schemas match. SceneMind-X does not install or start remote models automatically.

### 3. Secure network access

Bind remote services to server-local `127.0.0.1` and use a VPN, SSH tunnel, or another controlled private link. This example contains placeholders only:

```powershell
ssh -N -L <LOCAL_VLM_PORT>:127.0.0.1:<REMOTE_VLM_PORT> -L <LOCAL_EMBEDDING_PORT>:127.0.0.1:<REMOTE_EMBEDDING_PORT> <SERVER_ALIAS>
```

In the same PowerShell session that starts SceneMind-X:

```powershell
$env:SCENEMINDX_VLM_ENDPOINT='http://127.0.0.1:<LOCAL_VLM_PORT>'
$env:SCENEMINDX_E1_EMBEDDING_ENDPOINT='http://127.0.0.1:<LOCAL_EMBEDDING_PORT>'
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_scenemindx.ps1
```

Never expose the service publicly or store private keys, tokens, real addresses, or accounts in the project.

### 4. Selection and Ready checks

1. Check `/health` on both local tunnel endpoints and confirm both remote models are loaded.
2. Start SceneMind-X and open model access.
3. Select the Self-hosted display option `服务器映射（课程演示）`.
4. Click `重新检查服务器连接`; verify VLM and Embedding separately report Ready and Embedding reports 2048D.
5. Save the selection, then run one VQA and one retrieval with existing images.

Capabilities remain separate: one unavailable service never makes the whole Provider falsely Ready. Self-hosted and Local share a model-space contract but maintain separate physical Faiss instances; Cloud 2560D cannot be mixed.

### 5. Stop and isolate

Stop only the tunnel and remote services you own. Never terminate unrelated shared-server projects by process name. The remote operator is responsible for isolating and backing up models, environments, logs, caches, and indexes.
