---
type: index
tags:
  - Memba
  - Language-Model
  - model-structure
  - Mamba
status: active
created: 2026-05-15
updated: 2026-05-15
authors: WeiTa
---

# Mamba 架構演化索引（線性脈絡 → 原子卡片深讀）

## 這份索引在整體筆記扮演的角色
在 [[Mamba 系列 MOC]] 中，`## 2. Mamba 架構演化脈絡` 的目的不是列舉論文版本，而是提供：

- **同一條主線**：Mamba-1/2/3 各自「為了什麼硬限制／瓶頸」而改動
- **最小必要的技術語彙**：每一代新增的關鍵結構到底在做什麼
- **可深讀的卡片入口**：想看公式、假設、代價與設計理念時能直接跳

讀完這份索引，你應該可以回答：
> Mamba 系列不是隨機堆技巧，而是在「表達能力 × 平行化 × 推論 memory-bound」三角形中逐代重新配置。

---

## 0) 問題總覽：Mamba 系列到底在解什麼問題？
Mamba 系列的演化，本質上是在持續嘗試解決以下矛盾：

| 目標                 | 困難點                 |
| ------------------ | ------------------- |
| 保留 SSM 的線性時間與長序列效率 | 傳統 SSM 缺乏內容感知能力     |
| 提升模型對語言的理解能力       | 動態化後難以平行化           |
| 充分利用現代 GPU         | 遞迴運算不適合 Tensor Core |
| 強化推理與狀態追蹤能力        | 更強模型通常代表更高推論成本      |

深讀：[[Mamba/Mamba - 01 核心矛盾：SSM 能力 × 硬體特性]]

---

## 1) 為什麼 S4 仍難取代 Transformer？（Mamba 的出發點）
一句話：S4 會「記憶」，但不會「選擇該記什麼」。

深讀：[[Mamba/Mamba - 02 起點：為什麼 S4 不夠？（內容感知）]]

---

## 2) Mamba-1：把「選擇性」引入 SSM
關鍵轉折：從 LTI 變成 input-dependent dynamics（$B_t, C_t, \Delta_t$）。

深讀：
- Selective SSM 機制（核心公式＋直覺）：[[Mamba/Mamba - 03 Mamba-1：Selective SSM（B_t,C_t,Δ_t）]]
- $\Delta_t$ 的過濾/更新直覺：[[Mamba/Mamba - 04 Mamba-1：Δ_t 作為資訊過濾與動態門控]]
- 代價：失去 FFT；取而代之的是 scan 平行化：[[Mamba/Mamba - 05 Mamba-1：為何失去 FFT？以及 Parallel Scan 的位置]]

---

## 3) Mamba-2：SSD（Structured State Space Duality）把計算拉回矩陣乘法
關鍵轉折：將部分 recurrence 重寫為矩陣形式以吃滿 Tensor Core。

深讀：
- SSD 的動機與限制：[[Mamba/Mamba - 06 Mamba-2：SSD 的問題設定與限制]]
- SSD 的矩陣形式與 chunk 混合策略：[[Mamba/Mamba - 07 Mamba-2：SSD 的矩陣形式與 Block-Chunk 策略]]

---

## 4) Mamba-3：Inference-oriented redesign（推論 memory-bound 的重新配置）
關鍵轉折：推論常是 memory-bound → 可以用閒置 ALU 換更強 dynamics。

深讀：
- 指數梯形離散化（讓 local mixing 隱含進 recurrence）：[[Mamba/Mamba - 08 Mamba-3：Exponential-Trapezoidal Discretization]]
- RoPE 與複數動力學直覺（週期/相位追蹤）：[[Mamba/Mamba - 09 Mamba-3：Complex-valued Dynamics via RoPE（直覺）]]
- MIMO / rank-$R$（提高 arithmetic intensity）：[[Mamba/Mamba - 10 Mamba-3：MIMO 與 memory-bound 觀點]]

---

## 5) 拓樸層面的精簡：H3 → Mamba-1/2 → Mamba-3
這一層關注「模塊怎麼被融合/移除」，例如：移除 MLP、顯式卷積變隱含卷積、BCNorm 的位置與目的。

深讀：[[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]

---

## 連回 SSM 理論地基
若中途忘記符號或離散化定義，回到：[[SSM (State Space Model)|SSM 索引]]
