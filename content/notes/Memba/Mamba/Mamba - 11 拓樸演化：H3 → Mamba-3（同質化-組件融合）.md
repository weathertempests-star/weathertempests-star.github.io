---
type: atomic
tags:
  - Memba
  - model-structure
  - Mamba
status: active
created: 2026-05-15
updated: 2026-05-15
authors: WeiTa
---

## 這張卡在整體脈絡扮演什麼角色
這張卡用「拓樸」而非「公式」的角度整理：哪些模塊被移除/融合，為什麼這能同時提升效率與穩定性。

它對應到 MOC 裡 `架構精簡設計` 這個入口：讀者想看的是「模塊怎麼變了」而不是推導細節。

---

## 核心演進邏輯：極致同質化與組件融合

Mamba 系列的架構演進目標在於簡化傳統神經網路的複雜堆疊，將多個功能模塊融合，以提升計算效率與參數利用率。其核心在於打破「SSM + MLP」的傳統範式。

> 關鍵詞：同質化（Homogenization）＝用同一個 block 同時承擔「序列建模 + 非線性映射」。

---

## 演化階段對照表

| 階段            | 關鍵組件                 | 非線性映射           | 局部特徵處理              | 歸一化策略        |
| ------------- | -------------------- | --------------- | ------------------- | ------------ |
| **H3 (早期)**   | SSM-Shift + SSM-Diag | 獨立 MLP 區塊       | SSM-Shift (滑動窗口)    | 傳統 LayerNorm |
| **Mamba-1/2** | 單一整合區塊               | 融入 GLU (門控線性單元) | 短因果卷積 (Causal Conv) | RMSNorm      |
| **Mamba-3**   | **指數梯形離散化 SSM**      | 深度融合 GLU        | **隱含卷積**（實體卷積移除）    | **BCNorm**   |

---

## 第一階段：H3 架構的組合設計
早期線性序列模型 H3（Hungry Hungry Hippos）採用功能分離的策略：
- **SSM-Shift**：專職處理短期記憶，本質上是滑動窗口卷積
- **SSM-Diag**：負責捕捉長期依賴，使用對角狀態矩陣
- **殘留冗餘**：架構中保留了 Transformer 式的龐大 **MLP** 區塊

---

## 第二階段：Mamba 的同質化革命
Mamba 透過「同質化」重構，大幅精簡資料流路徑：
- **移除 MLP**：取消獨立的 MLP 區塊，轉而將 SSM 層與 **GLU** 深度融合
- **整合優勢**：減少層級間的資料搬移（Data Movement），並使每一層參數同時具備序列建模與非線性映射能力

---

## 第三階段：Mamba-3 的極簡與穩定性突破

### 1) 移除實體卷積層（Explicit Conv → Implicit Conv）
在 Mamba-3 之前，模型通常在 SSM 外部放置短因果卷積來處理局部上下文。

但 Mamba-3 引入指數梯形離散化後，其 update 形式顯式混入前一個輸入項：

$$
h_t
=
\alpha_t h_{t-1}
+
\beta_t B_{t-1}x_{t-1}
+
\gamma_t B_tx_t
$$

直覺上這等價於「在 recurrence 裡自帶短距離 mixing」，因此局部卷積的角色可以被吸收，進而移除顯式卷積模塊。

（離散化細節見：[[Mamba/Mamba - 08 Mamba-3：Exponential-Trapezoidal Discretization]]）

---

### 2) BCNorm：動態參數的歸一化策略
Mamba-3 引入 **BCNorm**，用來補強線性遞迴在動態參數下的數值穩定性。

機制：模型在線性投影生成動態矩陣 $B$ 與 $C$ 後，立即做歸一化（此處沿用你原筆記描述）：

$$
\text{Project}(x) \rightarrow B, C \rightarrow \text{RMSNorm}(B), \text{RMSNorm}(C) \rightarrow \text{SSM Recurrence}
$$

功能上可類比 Transformer 的 QKNorm：在進入核心交互（attention / recurrence）前，先把動態量綁回穩定範圍，減少長序列下的不穩定累積。

---

## 下一步
- 若想用「一條 Mamba-1/2/3 主線」理解動機與代價：回到 [[Mamba/Mamba - 00 演化索引（線性脈絡）]]
