---
title: "BERT 嵌入 (Embedding)"
description: "將日誌文本語義化為高維向量，支援多種 BERT 變體"
tags: [stage-1, bert, embedding, nlp]
date: 2025-01-01
---

# BERT 嵌入

> **目標**：將離散的日誌文字轉換為連續語義向量，讓語義相近的日誌在向量空間中距離更近。

[[Preprocessing|← Stage I 總覽]]

---

## 處理流程

```
中間資料 CSV（Intermediate_data/）
          │
          ▼ Tokenization
┌──────────────────────────┐
│  BERT 模型（批次推論）    │
│  batch_size=32           │
└──────────────────────────┘
          │
          ▼ Mean Pooling
   embedding 向量 (768-dim)
          │
          ▼ 可選 L2 Normalize
data/Embeddings/{id}_embeddings/
```

---

## 支援模型

| Model Key | 類型 | 維度 | 說明 |
|-----------|------|------|------|
| `secbert` | TransformerBERT | 768 | **推薦**，資安術語理解最佳 |
| `sentence-bert` | SentenceBERT | 384 | 快速，通用語義相似度 |
| `sentence-bert-large` | SentenceBERT | 768 | 高品質通用嵌入 |
| `bert-base` | TransformerBERT | 768 | 原始 BERT |
| `cti-bert` | SentenceBERT | 384 | 威脅情報專用 |

---

## 使用方式

```python
from models.bert import get_bert_model

bert = get_bert_model('secbert')  # 自動載入

embeddings = bert.embed(
    ["CreateFile C:\\test.exe SUCCESS",
     "Suspicious PowerShell detected"],
    batch_size=32,
    normalize=True   # L2 正規化，便於餘弦相似度
)
# embeddings.shape → (2, 768)
```

---

## 新增自訂模型

在 `models/bert.py` 的 `MODEL_REGISTRY` 中新增一行：

```python
MODEL_REGISTRY = {
    # ...既有模型...
    'my-model': {
        'class': TransformerBERTModel,
        'model_name': 'org/model-name',   # HuggingFace Model ID
        'description': '自訂模型 (768 dim)'
    },
}
```

無需修改其他程式碼，系統會自動註冊。

---

## 雙流嵌入設計（啟用解析器時）

```
Template  → BERT → Attention → Context Vector ┐
                                               ├→ 串接 → Log Vector
Parameters → BERT → Attention → Context Vector ┘
```

分離嵌入讓系統能分開處理「日誌結構」與「具體內容」的語義。

---

## 關鍵超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `BERT_MODEL_NAME` | `securebert2` | 嵌入模型選擇 |
| `batch_size` | `32` | 批次大小 |
| `normalize` | `True` | L2 正規化 |
| `BILSTM_HIDDEN_SIZE` | `128` | BiLSTM 隱藏層維度 |
| `BILSTM_NUM_LAYERS` | `2` | BiLSTM 層數 |

---

## 相關筆記

- [[Preprocessing|Stage I 預處理總覽]]
- [[TF-IDF|TF-IDF 三層架構]] — 與嵌入互補的詞彙匹配
- [[stage-3/Concept-Extraction|概念提取 NMF]] — 消費此嵌入向量的下游模組