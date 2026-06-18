---
title: "概念提取 (NMF)"
description: "Per-Dataset NMF 將高維嵌入映射至潛在概念空間，以 MITRE 作語義錨點"
tags: [stage-3, nmf, concept-extraction, dimensionality-reduction]
date: 2025-01-01
---

# 概念提取（ConceptExtractor）

> **目標**：將 768-dim 嵌入向量壓縮至 75-dim 概念空間，每個維度代表一個攻擊相關「概念」。

[[index|← 回到首頁]]

---

## 核心策略：Per-Dataset + 語義錨點

每個 Dataset 獨立訓練一個 NMF 模型，聯合 MITRE ATT&CK Technique 向量進行分解：

```
X_train = [X_dataset ; X_mitre]
              ↓ NMF
         H × W ≈ X_train
              ↓ 僅取 Dataset 部分
         H_dataset (N × 75)  ← 概念向量
```

**語義錨點的作用**：MITRE 向量引導 NMF 學習與已知攻擊模式相關的概念，使輸出的概念空間具有攻擊語義含義。

---

## 四步驟處理流程（Per-Dataset）

```
Step 1  載入外部知識（MITRE Embeddings）
   ↓
Step 2  載入該 Dataset 的 Log Vectors
   ↓
Step 3  聯合訓練 NMF（含 L1 稀疏約束）
         X_train = [X_dataset ; X_mitre]
   ↓
Step 4  僅轉換 Dataset 部分
         H_dataset → 儲存為 ConceptVectors
```

---

## NMF 數學細節

$$
X \approx H \cdot W \quad (X \in \mathbb{R}^{n \times 768}_+,\ H \in \mathbb{R}^{n \times 75}_+)
$$

### L1 稀疏更新規則

$$
H = H \cdot \frac{W^T X}{W^T W H + \lambda_{L1} + \epsilon}
$$

稀疏性約束使每個日誌傾向只屬於少數幾個明確概念，提升可解釋性。

---

## GPU 加速（NMFGpu）

使用 PyTorch 後端，支援乘法更新規則並行加速：

- **OOM 保護**：自動偵測 VRAM，動態計算安全 batch size
- **相容性檢查**：啟動時測試 CUDA kernel，不相容自動回退 CPU
- **併發控制**：`use_gpu=True` 時強制 `n_jobs=1`，避免多 Process 競爭 VRAM

---

## 關鍵超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `NMF_COMPONENTS` | `75` | 概念數量 $k$ |
| `NMF_L1_REG` | `0.01` | L1 稀疏強度 |
| `NMF_MAX_ITER` | `500` | 最大迭代次數 |
| `NMF_INIT` | `nndsvd` | 初始化策略 |
| `NMF_USE_GPU` | `True` | GPU 加速開關 |

---

## Pipeline 呼叫模式

```python
extractor = ConceptExtractor(n_concepts=75)
extractor.load_external_knowledge("data/ExternalKnowledge")

for dataset_id, input_path in datasets:
    extractor.model = None          # 每個 Dataset 重置模型
    extractor._is_fitted = False
    concept_vectors = extractor.process_single_dataset(
        dataset_id=dataset_id,
        input_path=input_path,
        output_dir="data/ConceptVectors",
    )
```

---

## 輸出格式

| 輸出 | 路徑 |
|------|------|
| 概念向量 | `data/ConceptVectors/{id}_concepts/data.arrow` |
| NMF 模型 | `data/ConceptVectors/{id}_concepts/nmf_model.pkl` |

---

## 相關筆記

- [[Sequence-Clustering|序列分群 HMM]] — 消費此概念向量的下游模組
- [[stage-1/Embedding|BERT 嵌入]] — 概念提取的輸入來源
- [[stage-2/PCA-GMM|PCA + GMM]] — 同為降維，目標不同