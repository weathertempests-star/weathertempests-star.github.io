---
title: "TF-IDF 三層架構"
description: "詞彙指紋建立、混合評分與雙高加分機制"
tags: [stage-1, tfidf, scoring]
date: 2025-01-01
---

# TF-IDF 三層架構

> **設計動機**：BERT 捕捉語義相似性，TF-IDF 確保關鍵詞彙精確匹配，兩者互補。

[[Preprocessing|← Stage I 總覽]]

---

## 三層架構概覽

```
Layer 1：Reference TF-IDF（Stage I）
  輸入：MITRE ATT&CK 技術描述
  輸出：tfidf_vectorizer.pkl + mitre_tfidf_matrix.pkl
           ↓ 共享同一 Vectorizer
           ├─ Layer 2：Log TF-IDF（Stage I）
           │    輸入：原始日誌文本
           │    輸出：tfidf.npz（per dataset）
           │
           └─ Layer 3：Sequence TF-IDF（Stage III）
                輸入：HMM Cluster 聚合文本
                輸出：Sequence 指紋 → Cosine Similarity
```

> [!important] 向量空間一致性
> 三層共用同一個 `TfidfVectorizer`，確保 Log 與 MITRE 在相同空間比對。

---

## Layer 1：Reference TF-IDF

```python
from precompute_log_tfidf import build_reference_tfidf

vectorizer = build_reference_tfidf(
    force_rebuild=False,
    max_features=5000,   # 詞彙表大小
)
```

**輸出位置**：`data/ExternalKnowledge/MITRE_TFIDF/`

---

## Layer 2：Log TF-IDF

```python
from precompute_log_tfidf import compute_log_tfidf

stats = compute_log_tfidf(vectorizer=vectorizer)
# stats: {"success": 10, "skipped": 5, "failed": 0}
```

**輸出位置**：`data/Embeddings/{dataset_id}_embeddings/tfidf.npz`

---

## Layer 3：Sequence TF-IDF（Stage III 使用）

```python
# 聚合 Cluster 內所有日誌文本，再計算相似度
cluster_tfidf = vectorizer.transform(cluster_texts)
tfidf_similarities = cosine_similarity(cluster_tfidf, mitre_tfidf_matrix)
```

---

## 混合評分公式

$$
\text{Similarity} = w_{emb} \cdot \text{Sim}_{emb} + w_{tfidf} \cdot \text{Sim}_{tfidf} + \text{Boost}
$$

| 權重參數 | 預設值 | 說明 |
|---------|--------|------|
| `LABELING_WEIGHT_EMBEDDING` | `0.6` | Embedding 相似度 |
| `LABELING_WEIGHT_TFIDF` | `0.3` | TF-IDF 相似度 |
| `LABELING_DUAL_BOOST_WEIGHT` | `0.1` | 雙高加分 |

---

## 雙高加分機制

$$
\text{Boost} = \begin{cases} 0.1 \times \min(\text{Sim}_{emb}, \text{Sim}_{tfidf}) & \text{若兩者} \geq 0.5 \\ 0 & \text{否則} \end{cases}
$$

| 情境 | Embedding | TF-IDF | Boost |
|------|-----------|--------|-------|
| 雙高 | ≥ 0.5 | ≥ 0.5 | +0.1×min |
| 單高 Emb | ≥ 0.5 | < 0.5 | 0 |
| 單高 TF-IDF | < 0.5 | ≥ 0.5 | 0 |
| 雙低 | < 0.5 | < 0.5 | 0 |

---

## 配置參數

| 參數 | 預設值 |
|------|--------|
| `LABELING_ENABLE_DUAL_BOOST` | `True` |
| `LABELING_DUAL_BOOST_THRESHOLD` | `0.5` |
| `MITRE_TFIDF_DIR` | `data/ExternalKnowledge/MITRE_TFIDF` |

---

## 相關筆記

- [[Preprocessing|Stage I 預處理總覽]]
- [[Embedding|BERT 嵌入]] — Similarity Score 的另一半
- [[stage-3/Auto-Labeling|自動標註]] — 最終使用混合評分的地方