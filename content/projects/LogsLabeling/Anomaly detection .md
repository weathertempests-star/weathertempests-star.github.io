---
title: "Stage II：異常偵測總覽"
description: "四模型 Ensemble + MAD 自適應閾值 + 時間序列平滑化"
tags: [stage-2, anomaly-detection, ensemble]
date: 2025-01-01
---

# Stage II：異常偵測總覽

> **目標**：對日誌嵌入向量進行無監督異常分析，輸出 0~1 的 Anomaly Score。

[[index|← 回到首頁]]

---

## 四種偵測模型

| 模型 | 偵測視角 | 計算特性 |
|------|---------|---------|
| [[Isolation-Forest\|Isolation Forest]] | 幾何隔離（路徑長度） | $O(n \log n)$，高維適用 |
| [[COPOD\|COPOD]] | 統計機率（ECDF 尾端） | $O(nd)$，無超參數 |
| [[AutoEncoder\|AutoEncoder]] | 重建誤差（非線性學習） | 需 GPU 訓練，語義敏感 |
| [[PCA-GMM\|PCA + GMM]] | 密度估計（負對數似然） | 多模態行為建模 |

---

## 五階段處理流程

```
Stage 1  四模型並行偵測 → 原始異常分數
    ↓
Stage 2  分數正規化（MinMax / Rank / Z-Score）
    ↓
Stage 3  Ensemble 加權融合
          Score = Σ(weight_i × score_i)
    ↓
Stage 4  時間序列平滑化（可選）
          移動平均 / 中位數 / 高斯濾波
    ↓
Stage 5  MAD 閾值決策 → 0/1 標籤
```

---

## Ensemble 策略

$$
\text{Ensemble Score} = \sum_{i} w_i \times \text{NormalizedScore}_i
$$

預設各模型權重均為 `0.25`，可在 `ENSEMBLE_WEIGHTS` 中調整。

> [!tip] 互補性
> 四種模型基於不同假設（距離、密度、重建、分布），可捕捉不同類型的異常。

---

## MAD 自適應閾值（推薦）

$$
\text{MAD} = \text{median}(|score - \text{median}(score)|)
$$

$$
\text{閾值} = \text{median} + k \times \text{MAD} \times 1.4826
$$

| k 值 | 約等異常率 | 適用情境 |
|------|----------|---------|
| 2.5 | ~1% | 寬鬆偵測 |
| **3.0** | **~0.3%** | **推薦（平衡）** |
| 3.5 | ~0.05% | 嚴格偵測 |

---

## 參數快速調整指南

| 情境 | 建議調整 |
|------|---------|
| 誤報過多 | `MAD_THRESHOLD_MULTIPLIER=3.5`、`SMOOTHING_METHOD="median"` |
| 漏報過多 | `MAD_THRESHOLD_MULTIPLIER=2.5`、提高 COPOD 權重 |
| 低延遲需求 | 僅用 `["isolation_forest", "copod"]`、停用平滑化 |
| 少樣本（<100） | 僅用 `["isolation_forest"]`、改用 Percentile 閾值 |

---

## 輸出欄位

```python
{
    "{model}_raw_score": float,   # 各模型原始分數
    "{model}_score": float,       # 正規化分數
    "{model}_label": int,         # 0/1 標籤
    "ensemble_score": float,      # 最終 Ensemble 分數
    "ensemble_label": int         # 最終標籤（主要使用）
}
```

---

## 使用範例

```python
from anomaly_dection import run_detection

result = run_detection(
    input_dir="data/Embeddings",
    output_dir="data/Detection_Results",
    generate_viz=True,
    verbose=True
)
```

---

## 相關筆記

- [[Isolation-Forest|Isolation Forest]] — 幾何隔離
- [[COPOD|COPOD]] — 統計機率
- [[AutoEncoder|AutoEncoder]] — 重建誤差
- [[PCA-GMM|PCA + GMM]] — 密度估計
- [[stage-3/Auto-Labeling|自動標註]] — 消費 Anomaly Score 的下游模組