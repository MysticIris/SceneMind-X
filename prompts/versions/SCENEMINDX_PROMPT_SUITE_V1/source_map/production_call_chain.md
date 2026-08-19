# Prompt Suite V1 production call chain

本文件记录 Phase 5.1 真实运行代码如何装载、拼接、调用和解析 V1 Prompt。归档没有
改动任何调用函数、Prompt、Schema、Guard、parser 或生成参数。

## 公共消息与生成流程

`PersistentQwen3VLService._generate_raw` 为每个请求构造一个 `user` 消息，消息内容顺序
为请求中的全部图片（保持请求顺序），随后是 resolved Prompt 文本。没有独立的
system message。处理器使用 `apply_chat_template(tokenize=true,
add_generation_prompt=true, return_dict=true, return_tensors="pt")`，随后调用
Qwen3-VL 的 `generate`。公共设置和各任务 token 上限见
`../generation_settings/generation_settings.json`。

## P3 v1.4

1. `prompts/gate1/p3_registry.json` 将默认核心 Prompt 指向 `p3_v1_4`。
2. `PersistentQwen3VLService.analyze_image` 按 registry 顺序依次执行八个 stage。
3. 每个 stage 直接接收同一张原图和该 stage 的完整 Prompt，不接收前一 stage 输出。
4. `_extract_json_object` 解析结果；单字段 stage 允许
   `raw_text_fallback`，之后把八阶段结果拼成九字段对象。
5. `jsonschema.Draft202012Validator` 对
   `gate1_d3_semantic_review_payload_p3_v1_1` 校验。Schema 失败被记录到
   `schema_valid/schema_errors`，但顶层 `ServiceResult.status` 仍为 `success`。

当前调用函数：`src/scenemindx/services/vlm.py` 的
`PersistentQwen3VLService.analyze_image`。

## Description V1

1. 从 Phase 1 registry 读取
   `natural_chinese_detailed_description_v1.txt`。
2. 使用 `str.format` 依次注入 `facts` JSON 和 `options` JSON。
3. 调用输入为原图 + resolved Prompt；因此 P3 facts 是辅助证据而不是图像的替代。
4. 输出经过元文本清理、纯文本规范化和 OpenCC t2s。
5. Guard 只检查简体中文字符数是否为 150–350；没有独立 verified-text Guard。

当前调用函数：`PersistentQwen3VLService.describe_image`。这解释了 `114.jpg` 中 P3
报告“不可辨文字”但 Description 仍生成具体招牌文字的现有边界。

## VQA V1

1. 从 `vqa_v1.txt` 读取完整模板。
2. 依次注入 `question` 和 `evidence` JSON。
3. evidence 可含当前 P3 facts、OCR 状态/候选，以及同一资产会话的历史。
4. 原图和 resolved Prompt 一起发给模型；`_extract_json_object` 解析返回。
5. 运行时没有 JSON Schema validator；资产绑定和会话隔离由产品/API 层维护。

当前调用函数：`PersistentQwen3VLService.answer_question`。

## Compare V1

1. 从 `compare_v1.txt` 读取基础 Prompt。
2. 有用户 instruction 时，代码按顺序追加：
   - `图片N=asset_filename` 输入映射；
   - 用户比较或排序要求；
   - 必须逐张引用文件名、不得遗漏，以及可选 ranking 字段要求。
3. 所有图片按请求顺序发送；`_extract_json_object` 解析模型输出。
4. P3 facts 不进入模型 Prompt，而是由 API 作为确定性的 `rows` 附加到响应。

基础 Prompt 的字节副本位于 `../prompts/compare_v1.txt`，动态追加文本的规范化提取
位于 `../prompts/compare_dynamic_suffix_v1.txt`。当前调用函数：
`PersistentQwen3VLService.compare_images` 的非 ranking 分支。

## Rank V1

1. API 用 `RANK_CONTRACT:` 前缀进入同一 `compare_images` 函数的 ranking 分支。
2. 动态 Prompt 包含 position-to-asset 映射、N 个整数的全排列约束和用户 criterion。
3. 模型只能输出 ASCII 逗号分隔的图片位置序列。
4. `_parse_rank_positions` 验证 1..N 每个位置恰好一次。
5. API 再校验 asset/rank 集合，并用当前 P3 facts 确定性生成 reason。

因此它是“VLM Prompt 排序 + P3 确定性 reason”基线，不是训练后的 Reranker。
规范化提取位于 `../prompts/rank_contract_v1.txt`。

## Content V1

1. 从 `content_v1.txt` 读取完整模板。
2. 依次注入所有图片的 `facts` JSON 数组与 `options` JSON。
3. 原图按请求顺序与 resolved Prompt 一起输入。
4. `_extract_json_object` 解析模型输出。
5. `fact_guard` 是模型自己写出的字段，不是独立运行时 Guard。

当前调用函数：`PersistentQwen3VLService.generate_content`。

## 远端运行版本一致性

在冻结前按照本地私有服务器操作规范进行过一次只读核对。当前运行 app 目录指向
release `phase5_1_restore_20260730_142112_wtbytes`。P3/Phase 1 registry、四个
独立任务 Prompt、P3 Schema、`vlm.py` 与远端 runtime 的关键 SHA 均与本地发布源
一致。核对未上传、写入、重启或切换 release，也未在本归档中记录服务器端点、
端口、用户名或密钥信息。
