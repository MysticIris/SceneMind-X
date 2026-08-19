# SceneMind-X Local 部署｜视觉与自然语言处理课程项目

## 中文版

### 1. 定位

Local 是具备 NVIDIA GPU 时的可选本机部署方式。视觉与自然语言处理课程评阅首选 Cloud API；Local 适合希望让模型权重和推理留在本机的使用者。程序按需加载 VLM 与 Embedding，支持环境预检、分别加载和卸载，不要求两种模型同时常驻。

### 2. 必需模型与官方来源

课程包不分发权重。当前真实配置由 `configs/providers/local_models_manifest.json` 定义：

- VLM：`Qwen/Qwen3-VL-4B-Instruct`；[ModelScope 官方页](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)；[Hugging Face 固定 revision](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17)。
- Embedding：`Qwen/Qwen3-VL-Embedding-2B`；[ModelScope 官方页](https://modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B)；[Hugging Face 固定 revision](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda)。

可使用 ModelScope CLI、Hugging Face CLI 或网页下载完整模型快照。必须保留配置、Tokenizer、Processor、索引文件和全部权重分片，不能只下载单个 `safetensors` 文件。VLM 权重约 8.9 GB，Embedding 权重约 4.3 GB。

### 3. 下载后的准确目录

无论下载工具生成什么缓存名称，最终都将完整模型目录重命名并放到项目根目录下：

```text
models/local/qwen3-vl-4b-instruct/
models/local/qwen3-vl-embedding-2b/
```

VLM 目录至少应包含 `config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json`、`model.safetensors.index.json` 和两个权重分片。Embedding 目录至少应包含 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json` 和 `model.safetensors`。目录名采用上述小写形式，必须与 manifest 的 `relative_path` 完全一致。

### 4. 环境、显存与磁盘

在 SceneMind-X 根目录安装：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-local-extra.lock.txt
```

需要支持 CUDA 的 PyTorch 与 NVIDIA GPU。16 GiB 显存可作为谨慎尝试下限，24 GiB 更适合单模型按需加载；若希望两种能力同时常驻，应预留约 18 GiB 以上可用显存并以页面 preflight 实测为准。建议 32 GB 系统内存、30 GB 以上可用磁盘。8 GiB 级显卡不适合双模型常驻，CPU-only 不适合交互式 4B VLM。

### 5. 新用户操作顺序

1. 按第 3 节放好两个模型目录。
2. 安装基础依赖与 Local 扩展依赖。
3. 双击 `start_scenemindx.bat`，或在项目根目录运行 `powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_scenemindx.ps1`。
4. 打开顶部模型接入面板，选择“当前电脑本地加载”。
5. 点击“检查本机环境”。页面会核对 CUDA、依赖、显存、manifest 和权重文件；缺项必须先修复。
6. 点击“开始按需加载”。显存低于建议值时，只有明确接受 OOM 风险才使用强制尝试选项。
7. VLM 状态显示 Ready 后，用已有图片执行一次问答；Embedding 状态显示 Ready 且为 2048D 后，检查或构建对应 Local 索引并执行一次检索。
8. 不使用时点击“卸载本地模型”释放显存。

Local 与 Self-hosted 采用相同的 Qwen3-VL-Embedding-2B 2048D 表示合同，但维护独立物理 Faiss；不能与 Cloud 2560D 索引混用。当前代码不自动量化，也不要求额外 Reranker 权重。

---

## English Version

### 1. Positioning

Local is optional for users with an NVIDIA GPU. Cloud API is recommended for grading the Visual and Natural Language Processing course project. Local keeps weights and inference on the user's machine and loads VLM and Embedding on demand.

### 2. Required models and official sources

Weights are not bundled. `configs/providers/local_models_manifest.json` is authoritative:

- VLM: `Qwen/Qwen3-VL-4B-Instruct`; [official ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct); [pinned Hugging Face revision](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17).
- Embedding: `Qwen/Qwen3-VL-Embedding-2B`; [official ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B); [pinned Hugging Face revision](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda).

Download complete snapshots with ModelScope CLI, Hugging Face CLI, or the official web pages. Preserve configs, tokenizer/processor files, indexes, and every weight shard. The VLM weights are about 8.9 GB and the Embedding weights about 4.3 GB.

### 3. Exact directories

Rename downloaded snapshots if necessary and place them exactly at:

```text
models/local/qwen3-vl-4b-instruct/
models/local/qwen3-vl-embedding-2b/
```

The VLM directory needs the config, generation config, tokenizer, preprocessor, weight index, and both weight shards. The Embedding directory needs its config, tokenizer, preprocessor, and `model.safetensors`. Lowercase directory names must match each manifest `relative_path`.

### 4. Environment and hardware

From the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-local-extra.lock.txt
```

Use CUDA-enabled PyTorch and an NVIDIA GPU. Treat 16 GiB VRAM as a cautious trial floor and prefer 24 GiB for on-demand single-model use. Keeping both resident should have roughly 18 GiB or more free VRAM and must pass preflight. Use about 32 GB system RAM and at least 30 GB free disk. An 8-GiB GPU is unsuitable for dual residency; CPU-only interactive 4B inference is impractical.

### 5. New-user workflow

1. Place both complete model directories as specified above.
2. Install base and Local-extra dependencies.
3. Double-click `start_scenemindx.bat`, or run `powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_scenemindx.ps1` from the project root.
4. Select `当前电脑本地加载` in model access.
5. Click `检查本机环境` and resolve every dependency, CUDA, VRAM, manifest, or weight error.
6. Click `开始按需加载`; force a low-memory attempt only when accepting OOM risk.
7. After VLM is Ready, run one image question. After Embedding is Ready and reports 2048D, inspect/build the matching Local index and run one retrieval.
8. Click `卸载本地模型` when finished.

Local and Self-hosted share the Qwen3-VL-Embedding-2B 2048D representation contract but maintain independent physical Faiss instances. Neither can be mixed with Cloud 2560D. No automatic quantization or extra Reranker weights are required.
