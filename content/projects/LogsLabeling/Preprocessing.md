---
title: "Stage I：預處理總覽"
description: "日誌解析、BERT 嵌入、TF-IDF 的統一入口"
tags: [stage-1, preprocessing, pipeline]
date: 2025-01-01
---

# Stage I：預處理總覽

> **目標**：將原始 CSV 日誌轉換為可供異常偵測使用的嵌入向量與 TF-IDF 特徵。

[[index|← 回到首頁]]

---

## 統一入口 API

```python
from preprocess import process_all_inputs

result = process_all_inputs(
    n_datasets=10,
    enable_parser=False,       # 是否啟用模板化解析
    model_name="securebert2",  # BERT 模型
    enable_tfidf=True,         # 同步計算 TF-IDF
    verbose=True
)
```

---

## 三步驟流程

```
data/input_logs/*.csv
        │
        ▼ Step 1
┌─────────────────────┐
│   日誌模板化         │  → data/Intermediate_data/
│  (LogLoader)        │
└─────────────────────┘
        │
        ▼ Step 2
┌─────────────────────┐
│   BERT 嵌入          │  → data/Embeddings/
│  (LogEmbedder)      │
└─────────────────────┘
        │
        ▼ Step 3
┌─────────────────────┐
│   TF-IDF 處理        │  → data/ExternalKnowledge/MITRE_TFIDF/
│  (TfidfPipeline)    │     data/Embeddings/{id}/tfidf.npz
└─────────────────────┘
```

---

## 輸出資料格式

| 步驟 | 輸出路徑 | 欄位 |
|------|---------|------|
| 解析（啟用） | `data/Intermediate_data/{id}.csv` | `LogID`, `Template`, `Parameters`, `OriginalLog` |
| 解析（停用） | `data/Intermediate_data/{id}.csv` | `LogID`, `ConcatenatedLog` |
| 嵌入 | `data/Embeddings/{id}_embeddings/` | `LogID`, `embedding` |
| TF-IDF | `data/Embeddings/{id}_embeddings/tfidf.npz` | scipy sparse matrix |

---

## 關鍵超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `PREPROCESS_ENABLE_PARSER` | `False` | Pipeline 層級解析開關 |
| `DEFAULT_PARSER` | `"drain"` | 解析演算法 |
| `BERT_MODEL_NAME` | `"securebert2"` | 嵌入模型 |
| `SEQUENCE_WINDOW_SIZE` | `5` | BiLSTM 滑動視窗 |
| `SEQUENCE_STRIDE` | `3` | 滑動步長 |

---

## 相關筆記

- [[Templatize|日誌模板化]] — 解析器詳細機制
- [[Embedding|BERT 嵌入]] — 模型選擇與擴展
- [[TF-IDF|TF-IDF 三層架構]] — 詞彙指紋建立
- [[Add-Parser|撰寫新解析器]] — 插件式開發指南