# SceneMind-X Prompt Suite V1

`SCENEMINDX_PROMPT_SUITE_V1` 是 Phase 5.1 当前生产提示词链的只读冻结副本，
状态为 `production_baseline_frozen`。它是未来 V2、V3 在相同样本、模型与生成
参数下进行比较的正式基线，不替代各 Prompt 已有的内部名称。

## 冻结范围

- P3 v1.4 八阶段结构化图像分析 Prompt；
- `NATURAL_CHINESE_DETAILED_DESCRIPTION_V1`；
- VQA v1；
- Compare v1 及运行时动态 suffix；
- Rank v1 的位置序列输出契约；
- Content v1；
- 实际输出契约、JSON Schema、生成设置、动态拼接顺序、调用链与代表案例。

`prompts/` 中的独立生产 Prompt 是源文件的字节级副本。Compare 动态 suffix 和
Rank 契约原本内嵌于 `src/scenemindx/services/vlm.py`，此处保存其规范化提取文本，
并在 `source_map/` 中记录来源和提取方式。`resolved_examples/` 中的内容在 Trace
未保存完整 resolved Prompt 时由冻结源静态还原，均标记为
`reconstructed_from_frozen_source`。

## 版本事实

- 代码基线：`9d09e3afca2a2898e1eed9778f565ca258de6546`
- 工作树补丁：三个已部署但未提交的链路修复；详见 `manifest.json`
- 模型：`Qwen/Qwen3-VL-4B-Instruct`
- revision：`ebb281ec70b05090aa6165b016eac8ec08e71b17`
- 核心 Prompt：P3 v1.4
- 运行时 JSON Schema 文件：
  `gate1_d3_semantic_review_payload_p3_v1_1.schema.json`

“P3 v1.4”是核心 Prompt 版本，不是 Schema 文件版本。运行时真实组合是 P3 v1.4
八阶段 Prompt 加上述 P3 输出 Schema。

## 使用规则

1. 禁止覆盖或原地修改已经冻结的版本目录。
2. 新版本必须建立新的 `SCENEMINDX_PROMPT_SUITE_V<N>` 目录。
3. 新版本必须保存完整 Prompt、哈希、模型与 revision、Schema、生成设置、测试
   manifest、原始输出、自动/人工评测、成功/失败案例、相对上一版本的差异和生产状态。
4. V2/V3 必须能在相同样本、模型、revision 和参数下与 V1 比较。
5. 本目录不包含课程原始图片、私有截图、服务器端点或密钥信息。

评测摘要见 `evaluation/prompt_suite_v1_metrics.md`，案例见
`representative_cases/`，私有截图和静态审阅页仅位于
`artifacts/phase5_1_private/prompt_suite_v1_baseline_20260730_170737/`。
