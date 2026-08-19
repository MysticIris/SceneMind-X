# SceneMind-X Cloud API 部署｜视觉与自然语言处理课程项目

## 中文版

### 1. 适用范围与前置条件

Cloud API 是“视觉与自然语言处理”课程评阅的首选方式。建议 Windows 10/11 x64、Python 3.11+（已验证 CPython 3.13.9）、8 GB RAM（推荐 16 GB）、约 4 GB 可用磁盘、Edge/Chrome 和稳定互联网连接。本机不需要 NVIDIA GPU。

Cloud 标准档使用 `qwen3.6-flash`，高质量档使用 `qwen3.7-plus`，检索使用 `qwen3-vl-embedding` 2560D。调用会产生云端费用，并受地域、额度、权限和服务可用性影响。

### 2. 安装与启动

在解压后的项目根目录执行：

```powershell
python --version
python -m pip install -r requirements.txt
```

随后双击 `start_scenemindx.bat`，或执行：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_scenemindx.ps1
```

脚本只使用项目内相对路径，日志写入 `runtime/logs/`。等待终端显示：

```text
[SceneMind-X] SceneMind-X is ready.
```

若浏览器未自动打开，访问 `http://127.0.0.1:8765`。健康检查地址为 `http://127.0.0.1:8765/health/live`。

### 3. 配置自己的 API Key

公开仓库不包含任何真实凭据。请在模型接入面板选择“使用自己的阿里云百炼 API Key”，并仅在页面输入框中填写自己的 Key。个人 Key 只保存在当前后端运行会话中；不要在终端、截图、Markdown、聊天或 Git 提交中展示完整 Key。

`.secrets/CREDENTIAL_FILE_GOES_HERE.txt` 仅用于说明本地凭据文件的放置位置，不包含有效 Key；`.secrets/*.csv` 已被 Git 忽略。

### 4. 申请自己的阿里云百炼 API Key

2026 年 8 月核对的官方入口：

- API Key 官方说明：<https://help.aliyun.com/zh/model-studio/get-api-key>
- Base URL/地域说明：<https://help.aliyun.com/zh/model-studio/base-url>

操作流程：

1. 打开官方 API Key 页面并登录阿里云账号。
2. 按页面提示开通阿里云百炼（Model Studio）。
3. 进入百炼控制台的 API Key 管理页，选择与计划使用地域一致的业务空间；课程系统当前支持 `cn-beijing`。
4. 点击创建 API Key；推荐使用默认业务空间，除非已有明确 Workspace 管理需求。
5. 复制新 Key 并安全保存。控制台可能只在创建时完整显示一次。
6. 启动 SceneMind-X，打开顶部模型接入面板。
7. 选择“百炼云端模型”“使用自己的阿里云百炼 API Key”。
8. 在 `API Key` 输入框粘贴 Key，地域选择“华北2（北京） · cn-beijing”。
9. Endpoint 模式优先选择“北京地域共享兼容端点（推荐）”。仅在明确使用 Workspace 或自定义 Host 时填写相应字段。
10. 点击“测试 VLM 与 Embedding”。两项通过后点击“保存接入选择”，再点击“刷新连接状态”。

个人 Key 只保存在当前后端运行会话中；重启后需再次输入。若要替换课程文件中的默认 Key，应保持原 CSV 字段结构，修改后重新启动，不要在 README 中记录 Key。

### 5. 首次进入与 Provider 检查

1. 点击顶部模型接入区域。
2. 选择“云端标准 · qwen3.6-flash”；复杂多图或长内容可改用“云端高质量 · qwen3.7-plus”。
3. 点击“测试 VLM 与 Embedding”，确认模型 ID、地域和 2560D Embedding 状态。
4. 点击“检查云索引状态”。公开版 Train/Validation 基础 Cloud Faiss 索引为 0 项；Default Library 保留 88 项 overlay，新导入图片继续使用独立 overlay。
5. 点击“保存接入选择”和“刷新连接状态”。

### 6. 关闭

浏览器关闭不会自动停止后端。回到启动窗口，按 **Enter** 或 **Ctrl+C**，启动器会安全停止本次创建的后端。只停止本次启动器拥有的服务，不要按进程名批量停止其他 Python 进程。

---

## English Version

### 1. Scope and prerequisites

Cloud API is recommended for grading the Visual and Natural Language Processing course project. Use Windows 10/11 x64, Python 3.11+ (validated with CPython 3.13.9), at least 8 GB RAM (16 GB recommended), about 4 GB free disk space, Edge/Chrome, and stable internet. A local NVIDIA GPU is not required.

The standard tier uses `qwen3.6-flash`, the high-quality tier uses `qwen3.7-plus`, and retrieval uses the 2560D `qwen3-vl-embedding`. Calls incur cloud cost and remain subject to region, quota, permissions, and service availability.

### 2. Install and start

From the extracted project root:

```powershell
python --version
python -m pip install -r requirements.txt
```

Double-click `start_scenemindx.bat`, or run:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_scenemindx.ps1
```

All paths are package-relative and logs go to `runtime/logs/`. Wait for:

```text
[SceneMind-X] SceneMind-X is ready.
```

If no browser opens, visit `http://127.0.0.1:8765`. Liveness is available at `http://127.0.0.1:8765/health/live`.

### 3. Configure your own API key

The public repository contains no real credentials. In the model-access panel, select `使用自己的阿里云百炼 API Key` and enter your own key only in the page input. A personal key is held only in the current backend session. Never expose a complete key in a terminal, screenshot, Markdown, chat, or Git commit.

`.secrets/CREDENTIAL_FILE_GOES_HERE.txt` only describes where a local credential file may be placed and contains no valid key; `.secrets/*.csv` is ignored by Git.

### 4. Create your own Alibaba Cloud Model Studio API key

Official pages verified in August 2026:

- API key instructions: <https://help.aliyun.com/zh/model-studio/get-api-key>
- Base URL and region: <https://help.aliyun.com/zh/model-studio/base-url>

Steps:

1. Open the official API key page and sign in to Alibaba Cloud.
2. Activate Model Studio if required.
3. Open API Key management and select a workspace in the same region as the endpoint; SceneMind-X currently supports `cn-beijing`.
4. Create an API key. The default workspace is recommended unless you explicitly manage another Workspace.
5. Copy and store the key securely; the console may show it completely only once.
6. Start SceneMind-X and open the top model-access panel.
7. Select `百炼云端模型` and `使用自己的阿里云百炼 API Key`.
8. Paste the key, then select `华北2（北京） · cn-beijing`.
9. Prefer `北京地域共享兼容端点（推荐）`; use Workspace/custom Host only when required.
10. Click `测试 VLM 与 Embedding`, then `保存接入选择`, and finally `刷新连接状态`.

A personal key is session-only and must be entered again after a service restart. Never write the key in README or commit it to Git.

### 5. First Provider check

1. Open the model-access panel.
2. Select `云端标准 · qwen3.6-flash`; use `云端高质量 · qwen3.7-plus` for harder multi-image or long-form work.
3. Click `测试 VLM 与 Embedding` and verify model, region, and 2560D Embedding status.
4. Click `检查云索引状态`. The public Train/Validation Cloud Faiss base index contains 0 items; the Default Library retains an 88-item overlay, and newly imported assets continue to use the separate overlay.
5. Click `保存接入选择` and `刷新连接状态`.

### 6. Stop the service

Closing the browser does not stop the backend. Return to the launcher window and press **Enter** or **Ctrl+C** to safely stop the backend created by that launch. Never terminate unrelated Python processes by name.
