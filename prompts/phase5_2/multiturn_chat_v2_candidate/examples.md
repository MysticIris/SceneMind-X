# Phase 5.2-A Chat V2 Candidate examples

These examples define interface behavior only. They are not human Gold and do
not introduce new visual facts.

## Stable reference

Question: `IMG_2 和 IMG_1 有什么区别？`

```json
{
  "answer": "IMG_2 与 IMG_1 的主要差异是……",
  "image_references": ["IMG_2", "IMG_1"],
  "evidence": ["IMG_2：……", "IMG_1：……"],
  "uncertainty": []
}
```

## Evidence-limited refusal

Question: `IMG_1 里的人是谁？`

```json
{
  "answer": "仅凭这张图片无法可靠确认人物的真实身份。",
  "image_references": ["IMG_1"],
  "evidence": ["IMG_1 中可见人物，但没有可靠身份凭据。"],
  "uncertainty": ["人物真实身份无法从当前视觉证据确认。"]
}
```

Real asset IDs, filenames, SHA values, paths and URLs are intentionally absent.
