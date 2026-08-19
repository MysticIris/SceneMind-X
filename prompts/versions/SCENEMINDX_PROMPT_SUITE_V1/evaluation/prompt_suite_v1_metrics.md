# SceneMind-X Prompt Suite V1 baseline metrics

本表只整理既有 Phase 5.1 验收和 live restore 证据；本次归档新增模型请求为 0。
不同任务的规模、日期、审核方式和分母不同，因此不计算综合准确率。

| Prompt family | 样本量 | 工程成功率 | Schema/格式 | 人工语义通过率 | 主要失败 |
|---|---:|---:|---:|---:|---|
| P3 | 12 | 12/12（100.00%） | 10/12（83.33%）严格 Schema | 8/12（66.67%） | 2 个严格 Schema 失败；4 个语义失败；单字段 stage 可走 raw-text fallback |
| Description | 1 | 1/1（100.00%） | 1/1（100.00%）长度合同 | 0/1（0.00%） | `114.jpg` 在“不可辨文字”证据下仍生成 3 个具体文字断言 |
| VQA | 24 | 24/24（100.00%） | 24/24（100.00%）请求/解析完成；无独立 JSON Schema validator | 14/24（58.33%） | 位置、数量、文字和证据完整性 |
| Compare | 4 | 4/4（100.00%）返回全部输入 | 4/4（100.00%）资产覆盖/解析 | 2/4（50.00%） | 两组语义失败，历史上有 differences 空或弱的问题 |
| Rank | 4 | 4/4（100.00%）结构完整 | 4/4（100.00%）位置全排列 | 3/4（75.00%） | 一组文字密度排序失败；不是训练后 Reranker |
| Content | 13 | 13/13（100.00%）执行完成 | N/A — existing evidence does not provide a complete denominator | N/A — existing evidence does not provide a complete denominator | 1 个已知文字残差；模型自报 fact_guard 不是独立 Guard |

## 证据范围

- P3、VQA、Compare、Rank、Content 主表数据来自 2026-07-28 至 2026-07-29
  的 Phase 5.1 Train-only 冻结验收；来源为
  `docs/status/phase5_1_snapshot.json` 与
  `docs/experiments/phase5_1_product_usability_recovery_report.md`。
- P3、VQA、Compare、Rank 的人工分母完整；Content 没有完整的人工语义分母。
- VQA 另有 6/6（100.00%）当前图片绑定和 6/6（100.00%）独立会话证据。
- Description 行是 2026-07-30 对冻结生产 V1 的单样本 live restore/验收观察：
  请求 1/1、长度 1/1、人工语义 0/1。它不能外推为总体准确率。
- 所有样本均为 Train。归档未读取 Val、Test 或 Blind。
- `655.jpg`、`1058.jpg` 和 `114.jpg` 的既有真实 Trace 只用于案例归档，没有重跑。

## 检索说明

文本检索、`color-grid-v1` 图像基线和混合检索属于 Phase 5.1 产品背景，不属于
Prompt Suite 的模型 Prompt 评估，未计入任何 V1 准确率或综合分数。
