# Prompt Suite V1 representative success cases

以下案例均来自既有 Phase 5.1 live restore 的真实 Train Trace。本次归档没有重新调用
模型。Trace 当时未保存完整 resolved Prompt，因此本文引用
`resolved_examples/` 中由冻结源静态还原并明确标记为
`reconstructed_from_frozen_source` 的版本；不声称它们是从 Trace 原样提取。

## `655.jpg`：P3 v1.4 + 两轮 VQA

- 用户任务：
  1. 分析图像；
  2. 回答“画面中有几个主要动物主体，它们是什么？”；
  3. 在同一会话追问这些动物的关系和位置。
- 资产 SHA-256：
  `c8326e30d80e1da4d23d820c2d930caa8377c384e34e82578be780039f3a58d1`
- 使用 Prompt：P3 v1.4 八阶段 Prompt；VQA v1。
- resolved Prompt：
  - `../resolved_examples/p3_v1_4_655_resolved.json`
  - `../resolved_examples/vqa_v1_655_count_resolved.txt`
- 输入 facts/evidence：
  - P3：原图；
  - VQA：P3 当前 facts、OCR `not_available`/空候选、同一资产会话历史。
  - P3 核心事实为“两只猫咪依偎休息”“黑白猫与橘白猫依偎”“无文字”。
- 模型输出：
  - P3 识别两只猫依偎休息；
  - 首轮 VQA 回答“两只”，并将黑白猫定位在右侧、橘白猫定位在左侧；
  - 追问回答两只猫相互依傍，并保持相同的左右关系。
- parser 结果：
  - P3 九字段对象通过最终 JSON Schema；
  - VQA 两轮均解析为 `answer/direct_observations/ocr_evidence/inference/uncertainty`。
- 页面结果：图像、P3 事实、两轮回答、证据与会话 ID 均正常显示；切换资产时会话
  隔离，刷新后可恢复同一资产会话。
- 成功判定：数量、类别、关系和左右位置与可见图像一致，且多轮历史没有串到其他
  资产。
- Trace：
  - P3：`470cb095-b281-4856-8602-ee0ebb1eb854`
  - VQA 首轮：`2dfc25e5-afda-44c4-b131-05e3ffcaeab9`
  - VQA 追问：`756ab69b-75d2-4074-8daf-b7235603c57a`
- 私有截图：
  - `artifacts/phase5_1_private/live_restore_20260730_142112/screenshots/analysis_655_jpg.png`
  - `artifacts/phase5_1_private/live_restore_20260730_142112/screenshots/vqa_655_jpg.png`
- 次要问题：P3 的 attributes、relations、scene、uncertainty、evidence 五个单字段
  stage 使用了 `raw_text_fallback`。最终对象 Schema 有效且该样本语义正确，但这是
  应持续统计的工程格式残差。

### 已解析 VQA 结果摘录

```json
{
  "answer": "画面中有两个主要动物主体，均为猫咪。",
  "direct_observations": [
    "一只黑白相间的猫咪位于画面右侧，面部清晰可见。",
    "一只橘白相间的猫咪位于画面左侧，部分被遮挡但可辨识为另一只猫。"
  ],
  "ocr_evidence": [],
  "inference": [],
  "uncertainty": []
}
```

## `1058.jpg`：P3 v1.4 + 事实型 Content

- 用户任务：分析雪山湖泊图像，并面向课程项目审阅者生成约 120 字的客观事实内容。
- 资产 SHA-256：
  `25ee4eaaf09277f6e14e2a1bea6dac881987abe60fdaa669b3be0d3616ecc310`
- 使用 Prompt：P3 v1.4 八阶段 Prompt；Content v1。
- resolved Prompt：
  - `../resolved_examples/p3_v1_4_655_resolved.json` 展示 P3 八阶段调用形态；
    本样本的 stage Prompt 文本相同，输入资产不同。
  - `../resolved_examples/content_v1_1058_resolved.txt`
- 输入 facts/evidence：P3 给出“雪山映湖水”“夕阳染金”“天空在上方”“无文字”
  等当前 facts；Content options 为客观事实、课程项目审阅者、长度 120、factual。
- 模型输出：
  `图像呈现雪山倒映于湖面，夕阳余晖染金山巅，天空湛蓝无云。`
- parser 结果：JSON 对象成功解析，`used_facts` 对应雪山映湖、夕阳染金和天空；
  `fact_guard` 返回 `passed`。
- 页面结果：事实型模式、单图、生成内容、事实依据和 Trace 均正常显示。
- 成功判定：生成文本简洁，核心陈述可由 P3 facts 和原图支持，没有添加具体地名、
  人物或不可见文字。该图在同一 live restore 的 Compare 中也被正确归因于雪山湖泊
  自然场景。
- Trace：
  - P3：`af491fa9-afef-4eb2-bb11-9a8902123835`
  - Content：`654690f3-5e6e-469b-a3b3-5f4fcda48dde`
  - Compare 归因：`0b54d5c5-6e2f-4faf-98bc-c1b90a84c72a`
- 私有截图：
  - `artifacts/phase5_1_private/live_restore_20260730_142112/screenshots/analysis_1058_jpg.png`
  - `artifacts/phase5_1_private/live_restore_20260730_142112/screenshots/content_generation_1058_jpg.png`
  - `artifacts/phase5_1_private/live_restore_20260730_142112/screenshots/multi_image_compare.png`
- 次要问题：模型原始 JSON 把预期字段 `omitted_uncertain_facts` 拼成
  `omitted_uncertain_fasts`。当前 parser 仍接受对象，语义内容正确，但运行时没有
  独立 JSON Schema validator，且 `fact_guard` 只是模型自报。
