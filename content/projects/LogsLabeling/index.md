---
title: "日誌威脅獵捕 Pipeline"
description: "以無監督學習自動標註系統日誌中的 MITRE ATT&CK 攻擊技術"
tags: [overview, pipeline]
date: 2025-01-01
---

# 日誌威脅獵捕 Pipeline

> 將原始系統日誌轉化為 MITRE ATT&CK 技術標註，全程無需人工標籤。

本專案採用三階段非監督學習管線，對系統日誌進行解析、嵌入、異常偵測，最終自動對應至 MITRE ATT&CK 攻擊技術框架。

---

## Pipeline 三階段總覽

```
原始 CSV 日誌
     │
     ▼
┌─────────────────────────────┐
│   Stage I：輸入處理          │  解析 → 嵌入 → TF-IDF
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Stage II：異常偵測         │  四模型 Ensemble
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Stage III：自動標註        │  NMF → HMM → Labeling
└─────────────────────────────┘
     │
     ▼
  Labeled CSV（含 MITRE 技術 ID）
```

---

## 閱讀路徑

### 🔵 Stage I — 輸入處理

| 筆記 | 說明 |
|------|------|
| [[stage-1/Preprocessing\|預處理總覽]] | Stage I 統一入口與三步驟流程 |
| [[stage-1/Templatize\|日誌模板化]] | Drain / Spell / LenMa 解析器 |
| [[stage-1/Add-Parser\|撰寫新解析器]] | 插件式解析器開發指南 |
| [[stage-1/Embedding\|BERT 嵌入]] | 語義向量化與模型選擇 |
| [[stage-1/TF-IDF\|TF-IDF 三層架構]] | 詞彙指紋與混合評分基礎 |

### 🟠 Stage II — 異常偵測

| 筆記 | 說明 |
|------|------|
| [[stage-2/Anomaly-Detection\|異常偵測總覽]] | Ensemble 策略與 MAD 閾值 |
| [[stage-2/Isolation-Forest\|Isolation Forest]] | 幾何隔離，高維適用 |
| [[stage-2/COPOD\|COPOD]] | 統計機率，無超參數 |
| [[stage-2/AutoEncoder\|AutoEncoder]] | 重建誤差，語義學習 |
| [[stage-2/PCA-GMM\|PCA + GMM]] | 密度估計，多模態建模 |

### 🟢 Stage III — 自動標註

| 筆記 | 說明 |
|------|------|
| [[stage-3/Concept-Extraction\|概念提取 NMF]] | 高維向量 → 潛在概念空間 |
| [[stage-3/Sequence-Clustering\|序列分群 HMM]] | 攻擊階段識別 |
| [[stage-3/Auto-Labeling\|自動標註]] | MITRE 技術對應與威脅信心度 |

---

## 核心評分公式

$$
\text{Threat Confidence} = 0.7 \times \text{Similarity} + 0.3 \times \text{Anomaly Score}
$$

$$
\text{Similarity} = 0.6 \times \text{Sim}_{emb} + 0.3 \times \text{Sim}_{tfidf} + \text{Dual-High Boost}
$$

---

## Tags

`#pipeline` `#nlp` `#anomaly-detection` `#mitre-attack` `#unsupervised-learning`