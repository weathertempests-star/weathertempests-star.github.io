---
type: permanent
tags:
  - Memba
  - model-structure
created: 2026-05-14
updated: 2026-05-15
status: active
authors: WeiTa
---

# Mamba 架構的拓樸演化（索引入口）

## 這份筆記在 MOC 扮演什麼角色
這份筆記用「模塊拓樸」的視角，回答一個更工程導向的問題：
> 為什麼 Mamba-3 可以在移除更多顯式組件的同時，還變得更穩、更能長度外推？

若你想先把 Mamba-1/2/3 的動機串起來，再回來看拓樸：[[Mamba/Mamba - 00 演化索引（線性脈絡）]]

深讀（原子卡）：[[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]

---

## 核心演進邏輯：極致同質化與組件融合

Mamba 系列的架構演進目標在於簡化傳統神經網路的複雜堆疊，將多個功能模塊融合，以提升計算效率與參數利用率。其核心在於打破「SSM + MLP」的傳統範式。

關鍵詞可以先抓一個：
> 同質化（Homogenization）＝用同一個 block 同時承擔「序列建模 + 非線性映射」。


## 演化階段對照表

| 階段 | 關鍵組件 | 非線性映射 | 局部特徵處理 | 歸一化策略 |
| --- | --- | --- | --- | --- |
| **H3 (早期)** | SSM-Shift + SSM-Diag | 獨立 MLP 區塊 | SSM-Shift (滑動窗口) | 傳統 LayerNorm |
| **Mamba-1/2** | 單一整合區塊 | 融入 GLU (門控線性單元) | 短因果卷積 (Causal Conv) | RMSNorm |
| **Mamba-3** | **指數梯形離散化 SSM** | 深度融合 GLU | **隱含卷積** (實體卷積移除) | **BCNorm** |

---

## 第一階段：H3 架構的組合設計

早期線性序列模型 H3（Hungry Hungry Hippos）採用功能分離的策略：

* **SSM-Shift**：專職處理短期記憶，本質上是滑動窗口卷積。
* **SSM-Diag**：負責捕捉長期依賴，使用對角狀態矩陣。
* **殘留冗餘**：架構中保留了 Transformer 式的龐大 **MLP (前饋神經網路)** 區塊。

---

## 第二階段：Mamba 的同質化革命

Mamba 透過「同質化（Homogenization）」重構，大幅精簡了資料流路徑：

* **移除 MLP**：取消獨立的 MLP 區塊，轉而將 SSM 層與 **GLU (Gated Linear Unit)** 深度融合。
* **整合優勢**：減少層級間的資料搬移（Data Movement），並使每一層參數同時具備序列建模與非線性映射能力。

---

## 第三階段：Mamba-3 的極簡與穩定性突破

### 1. 移除實體卷積層 (Removal of Explicit Convolution)

在 Mamba-3 之前，模型必須在 SSM 外部放置「短因果卷積層」來處理局部上下文。

* **技術變革**：由於 Mamba-3 引入了「指數梯形離散化」，其數學形式已包含當前輸入 $x_t$ 與前一輸入 $x_{t-1}$ 的關聯。
* **結果**：該離散化機制賦予了 SSM **隱含的數據依賴卷積能力**，因此可徹底移除實體的卷積模塊。

對應的細節可以在兩張卡補齊：
- 離散化如何引入「隱含 local mixing」：[[Mamba/Mamba - 08 Mamba-3：Exponential-Trapezoidal Discretization]]
- 從拓樸角度看 explicit → implicit 的組件融合：[[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]

### 2. BCNorm：動態參數的歸一化策略

Mamba-3 引入了 **BCNorm** 機制，這是線性模型在應對數值穩定性上的關鍵補強。

* **機制細節**：模型在線性投影生成動態矩陣 $B$ 與 $C$ 後，立即應用 **RMSNorm** 進行歸一化。
* **設計對標**：此設計在功能上對標 Transformer 的 QKNorm。
* **數值韌性**：
* 有效抑制遞迴過程中由動態參數波動引發的不穩定。
* 顯著提升模型在**長度外推 (Length Extrapolation)** 任務中的表現，使模型能處理超出訓練長度的序列。



> **歸一化核心邏輯**：
> 
> $$\text{Project}(x) \rightarrow B, C \rightarrow \text{RMSNorm}(B), \text{RMSNorm}(C) \rightarrow \text{SSM Recurrence}$$
> 
> 

---

## 相容性：舊內容去哪裡了？
這份筆記保留「一頁看懂」的敘述與表格；更細的拆解與延伸直覺已整理到：
- [[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]

## 結論

Mamba-3 的架構演進證明了：透過更精確的數學離散化（梯形法）與策略性的歸一化（BCNorm），可以在移除更多物理組件的同時，獲得更強的長序列建模韌性。