---
title: "自動標註 (Auto Labeling)"
description: "將 HMM 分群結果對應至 MITRE ATT&CK 技術，計算 Threat Confidence"
tags: [stage-3, auto-labeling, mitre-attack, scoring]
date: 2025-01-01
---

# 自動標註（Auto Labeling）

> **目標**：將 HMM 序列分群結果與 MITRE ATT&CK 比對，為每筆日誌計算威脅信心度（Threat Confidence）。

[[index|← 回到首頁]]

---

## 雙層評分架構

```
Layer 1：Similarity Score（與 MITRE 技術的相似程度）
─────────────────────────────────────────────
Cluster Centroid ──┐
                   ├── Embedding Sim ──┐
MITRE Embedding ───┘                   │
                                       ├── Similarity Score
Sequence TF-IDF ───┐                   │
                   ├── TF-IDF Sim ─────┤
MITRE TF-IDF 指紋 ─┘                   │
                                       │
Dual-High Boost ───────────────────────┘

Layer 2：Threat Confidence（最終威脅信心度）
─────────────────────────────────────────────
Similarity Score (α=0.7) ─┐
                           ├── Threat Confidence ── Top-K Label
Anomaly Score (β=0.3) ────┘
```

---

## 評分公式

### Layer 1：Similarity Score

$$
\text{Similarity} = 0.6 \cdot \text{Sim}_{emb} + 0.3 \cdot \text{Sim}_{tfidf} + \text{Boost}
$$

$$
\text{Boost} = \begin{cases} 0.1 \times \min(\text{Sim}_{emb}, \text{Sim}_{tfidf}) & \text{若兩者} \geq 0.5 \\ 0 & \text{否則} \end{cases}
$$

### Layer 2：Threat Confidence

$$
\text{Threat Confidence} = 0.7 \times \text{Similarity} + 0.3 \times \text{Anomaly Score}
$$

> [!info] 設計理念
> 即使一個 Sequence 與 MITRE 技術相似度很高，若 Anomaly Score 低（正常行為），Threat Confidence 也會降低，有效減少誤報。

---

## Sequence TF-IDF 計算步驟

```python
# Step 1：聚合每個 Cluster 的所有日誌文本
for cluster_id in unique_clusters:
    cluster_text = " ".join([log_texts[i] for i where label==cluster_id])

# Step 2：TF-IDF 轉換（使用 Stage I 的 Reference Vectorizer）
cluster_tfidf = vectorizer.transform(cluster_texts)

# Step 3：計算與 MITRE 的餘弦相似度
tfidf_sim = cosine_similarity(cluster_tfidf, mitre_tfidf_matrix)
```

---

## 輸出格式

**主要標註結果** (`{dataset_id}_Labeled.csv`):

| 欄位 | 說明 |
|------|------|
| `original_idx` | 日誌原始索引 |
| `anomaly_score` | Stage II 異常分數 (0~1) |
| `groundtruth_tid` | Ground Truth 技術 ID |
| `predicted_technique_1_name` | Top-1 預測技術 |
| `predicted_technique_1_threat_confidence` | Top-1 威脅信心度 |
| `predicted_technique_K_*` | Top-K 相應欄位 |

**摘要檔案** (`{dataset_id}_Summary.csv`):

Borda Count 排名聚合所有 Cluster 的技術投票，分別輸出 Embedding-only / TF-IDF-only / Hybrid 的 Top-5。

---

## 配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LABELING_WEIGHT_EMBEDDING` | `0.6` | Embedding 權重 |
| `LABELING_WEIGHT_TFIDF` | `0.3` | TF-IDF 權重 |
| `LABELING_DUAL_BOOST_THRESHOLD` | `0.5` | 雙高觸發閾值 |
| `LABELING_SIMILARITY_WEIGHT` | `0.7` | α（Similarity 權重） |
| `LABELING_ANOMALY_WEIGHT` | `0.3` | β（Anomaly 權重） |
| `LABELING_SIMILARITY_THRESHOLD` | `0.3` | 低於此值標記 Benign |
| `LABELING_TOP_K` | `3` | 輸出候選技術數量 |

---

## 獨立執行

```python
from auto_labeling import AutoLabeler

labeler = AutoLabeler()
labeler.load_mitre_embeddings()
labeler.load_mitre_tfidf()

result = labeler.process_single_dataset(
    dataset_id="dataset_001",
    concept_vectors=concept_vectors,
    cluster_labels=cluster_labels,
    output_dir="result/Labeling_Results/",
    anomaly_scores=anomaly_scores,   # Stage II 結果（可選）
)
```

---

## 資料依賴關係

| 依賴 | 來源 Stage |
|------|-----------|
| MITRE Embedding + TF-IDF 指紋 | Stage I |
| Log Anomaly Score | Stage II |
| NMF 概念向量 | Stage III-a |
| HMM 分群標籤 | Stage III-b |

---

## 相關筆記

- [[Sequence-Clustering|序列分群 HMM]] — 分群標籤的來源
- [[Concept-Extraction|概念提取 NMF]] — 概念向量的來源
- [[stage-1/TF-IDF|TF-IDF 三層架構]] — 混合評分機制
- [[stage-2/Anomaly-Detection|Stage II 異常偵測]] — Anomaly Score 來源