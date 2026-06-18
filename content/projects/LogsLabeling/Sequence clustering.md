---
title: "序列分群 (HMM)"
description: "雙軌特徵 + Per-Dataset HMM 識別攻擊行為的演變階段"
tags: [stage-3, hmm, sequence-clustering, time-series]
date: 2025-01-01
---

# 序列分群（Sequence Clustering）

> **目標**：針對單一攻擊行為（Dataset），精確識別其內部演變階段（如：初始存取 → 執行 → 清理）。

[[index|← 回到首頁]]

---

## 雙軌特徵機制

解決 NMF 輸出不易收斂至高斯假設的核心設計：

```
X_raw（原始 NMF 輸出）
     │
     ├──── 訓練軌道 ────────────────────────────────────────────────┐
     │     Log-Transform → Percentile Clipping → Z-Score           │
     │     + 一階差分 Δ = x_t - x_{t-1}                           │
     │     X_augmented = [X_norm, Δ]  ← 餵入 HMM 訓練              │
     │                                                             ▼
     │                                                     HMM 模型訓練
     │                                                     Viterbi 解碼
     │                                                             │
     └──── 應用軌道 ────────────────────────────────────────────────┘
           X_raw 用於計算 Cluster Centroid
           → 與 MITRE 向量做餘弦相似度
```

---

## 一階差分的意義

$$
\Delta_t = x_t - x_{t-1}
$$

| 場景 | $\Delta$ 特徵 |
|------|--------------|
| 攻擊開始 | 概念權重突升 → 正向大幅差分 |
| 攻擊結束 | 概念權重驟降 → 負向大幅差分 |
| 穩定期 | 概念平穩 → 差分接近零 |

捕捉「變化率」使 HMM 能更精準識別階段轉換點。

---

## 三階段常態化（解決 HMM 收斂問題）

```
X_raw
  ↓ Log-Transform：log(1 + |X|)        # 壓縮數值範圍
  ↓ Percentile Clipping：[0.5%, 99.5%] # 去除極端值
  ↓ Z-Score：(X - μ) / (σ + ε)         # 標準化為均值0、標準差1
= X_norm
```

> [!warning] 收斂問題排查
> 若出現 "Model is not converging"，首先確認三階段常態化是否正確執行。

---

## Per-Dataset 訓練流程

```
Step 1  載入概念矩陣 X_raw，執行三階段常態化 + 一階差分
   ↓
Step 2  動態調整 K 範圍
         K_max = min(K_config, n_samples / 5)
         若 Var(X) < 1e-8 → 強制 K = K_min（變異度保護）
   ↓
Step 3  Grid Search K ∈ [2, K_effective]
         每個 K 執行 20 次隨機初始化
         選 BIC 最低的模型（平衡精度與複雜度）
   ↓
Step 4  Viterbi 解碼 → labels 序列
         嚴格驗證 len(labels) == len(X_raw)
   ↓
Step 5  存檔 labels.npy + model.pkl
```

---

## 模型選擇：BIC 而非 Log-Likelihood

$$
\text{BIC} = -2 \ln L + k \ln n
$$

使用 BIC（貝葉斯資訊量準則）防止模型無腦將 K 選至最大，平衡擬合度與模型複雜度。

---

## 關鍵超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `HMM_K_MIN / K_MAX` | `2 / 10` | 狀態數搜尋範圍 |
| `HMM_N_STARTS` | `20` | 每個 K 的隨機初始化次數 |
| `HMM_COVARIANCE_TYPE` | `diag` | 對角協方差，速度最快 |
| `HMM_MIN_COVAR` | `1e-3` | 防 Singular Matrix |
| `HMM_TOL` | `1e-4` | 收斂門檻 |
| `HMM_ENABLE_PARALLEL` | `True` | 並行測試不同種子 |

---

## 常見問題排查

| 問題 | 解法 |
|------|------|
| Model not converging | 確認三階段常態化是否執行 |
| K 值總選最大 | 檢查 `n_samples/5`；若仍如此，回到 NMF 調整 L1 |
| 訓練過慢 | 確認 `HMM_ENABLE_PARALLEL=True` |
| 大量 Warning | 正常，只要有成功的 K 值即可繼續 |

---

## 輸出格式

| 輸出 | 路徑 | 說明 |
|------|------|------|
| 分群標籤 | `labels.npy` | 整數序列，長度嚴格對應 X_raw |
| 模型檔案 | `model.pkl` | 含 `model`, `best_k`, `scaler_mean`, `scaler_std` |

---

## 相關筆記

- [[Concept-Extraction|概念提取 NMF]] — 此模組的輸入來源
- [[Auto-Labeling|自動標註]] — 消費 labels 的下游模組