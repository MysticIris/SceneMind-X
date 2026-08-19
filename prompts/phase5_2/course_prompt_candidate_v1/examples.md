# Phase 5.2 Course Prompt Candidate V1 Examples

这些示例只说明合同，不是人工 Gold，也不构成 Prompt A/B 或质量晋升证据。

## 成功示例：多图事实问答

输入：两张不同场景图片，问题要求分别说明主体。  
期望：`asset_references` 同时包含两个真实 asset_id，各自证据不串图；
`refused=false`。

## 成功示例：无共同事件的多图生成

输入：猫、雪山、城市夜景三张无共同事件证据的图片，类型为“朋友圈”。  
期望：使用组图或合集叙事，`cross_image_relation=creative_sequence` 或
`independent`；不得宣称三张图拍摄于同一天同一地点。

## 成功示例：图文排序

输入：文字“最适合表现宁静自然风景”与 3-5 张图。  
期望：全部 asset_id 恰好出现一次，rank 连续，最佳项与 `best_asset_id`
一致，并说明自然景观匹配点。

## 失败/边界示例：未核验文字

输入图片存在模糊招牌，但 verified_text 为空。  
期望：回答“无法可靠辨认具体文字”或在 unknowns 中记录；不得猜测店名、
地名或英文单词。

## 失败/边界示例：无图视觉提问

此情况由应用层直接返回帮助/拒答，不调用 Candidate，不让模型伪造视觉答案。
