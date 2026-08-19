# Prompt Suite V1 representative failure cases

以下失败均来自既有真实 Train Trace，没有为归档新增模型请求。它们是后续 V2 比较
必须保留的回归样本，但本阶段不设计或修改 V2。

## `114.jpg`：中文描述生成未验证文字

- 用户任务：基于图像和 P3 facts，生成 180 字、自然客观的中文详细描述。
- 资产 SHA-256：
  `344847bf4679974b1ede8f13a2dfc3dd5d5269173eaae9910a6b876208f8ff21`
- 使用 Prompt：`NATURAL_CHINESE_DETAILED_DESCRIPTION_V1`。
- resolved Prompt：
  `../resolved_examples/description_v1_114_resolved.txt`
  （`reconstructed_from_frozen_source`）。
- 输入 facts/evidence：
  - `global_observation`: 城市夜景高楼灯光秀
  - `subjects`: 高楼灯光投影表演
  - `relations`: 高楼与古建筑并存夜景
  - `visible_text_candidates`: **不可辨文字**
  - OCR：没有 verified-text 支持。
- 模型原始输出：请求和长度合同成功，但内容断言了“中国平安”“复悦茶美食馆”
  和 `TOKYO` 三个具体文字字符串，并进一步用 `TOKYO` 推断商业区国际化属性。
- parser 结果：元文本清理、纯文本规范化和 OpenCC t2s 正常；中文字符数 209，
  `length_contract_pass=true`。parser 不验证事实或文字证据。
- 页面显示：完整展示了上述描述，没有将三处未验证文字标为不确定或拦截。
- 合理预期：在 P3 明确为“不可辨文字”且无 verified-text/OCR 证据时，不应生成
  具体招牌或地名；至少应省略或明确不可辨。
- 失败类型：`unsupported_specific_text_claim`；人工语义失败。
- 初步根因：Description Prompt 同时看到原图和 P3 facts，并要求描述可靠文字，
  但生产链没有独立 verified-text 输入合同或运行时事实 Guard。该问题横跨
  Prompt + Evidence/Guard 边界，长度 Guard 无法发现。
- Trace：`a5475354-ed30-4f5f-a9f0-d90fb23a9924`
- 私有截图：
  `artifacts/phase5_1_private/live_restore_20260730_142112/screenshots/description_114_jpg.png`
- 后续比较指标：
  1. 在 `visible_text_candidates=不可辨文字` 且无 verified text 时，具体文字断言数；
  2. unsupported text claim 样本率；
  3. 人工事实/证据通过率；
  4. 长度合同与语义正确率必须分开报告。

### 原始输出中的失败片段

```text
与旁边标有“中国平安”的建筑形成鲜明对比……
部分店铺招牌如“复悦茶美食馆”清晰可见……
画面右下角隐约可见“TOKYO”字样……
```

## `114.jpg`：VQA 位置回答不完整

- 用户任务：在上一轮“主要场景和主体”问答之后追问：
  “这些主体位于画面的什么位置？请给出可见证据。”
- 使用 Prompt：VQA v1。
- resolved Prompt：
  `../resolved_examples/vqa_v1_114_position_resolved.txt`
  （`reconstructed_from_frozen_source`）。
- 输入 facts/evidence：
  - 当前 P3 facts 包含城市夜景、高楼灯光投影、古建筑、人群和“不可辨文字”；
  - OCR 为 `not_available`；
  - 同一资产会话历史含前一轮 Trace
    `dbfa0734-8c5f-42f0-95c4-16fc4ccb6fd2`。
- 模型原始输出：`answer` 仅为“高楼灯光秀”；observations 列出灯光图案、
  古建筑和地面行人，但没有明确回答高楼、古建筑与人群分别位于画面的哪一部分。
- parser 结果：JSON 对象结构解析成功，所有预期顶层字段均存在。
- 页面显示：页面正常显示回答、observations、证据来源和会话 ID；没有工程错误。
- 合理预期：直接回答各主体的位置（例如画面中央/背景、下方/前景等），并把每个
  位置与可见证据对应。
- 失败类型：`current_question_incomplete_position_answer`；HTTP/解析成功但人工语义
  与证据失败。
- 初步根因：VQA Prompt 要求证据分层，却没有强制逐项覆盖当前问题中的位置槽位；
  会话历史中的上一轮主体概括可能强化了重复概括而非回答新问题。
- Trace：`e31a851c-a703-4e2b-8cf8-6d6bbbd2250d`
- 私有截图：
  `artifacts/phase5_1_private/live_restore_20260730_142112/screenshots/vqa_114_jpg.png`
- 后续比较指标：
  1. position question slot coverage；
  2. 当前问题直接回答完整率；
  3. observation 与 answer 的逐项证据覆盖；
  4. 多轮中重复上一轮答案而遗漏新约束的比例。
