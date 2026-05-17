---
type: permanent
tags:
  - Memba
  - Language-Model
  - model-structure
status: active
created: 2026-05-14
updated: 2026-05-15
authors: WeiTa
---


# Mamba 架構演化脈絡（索引入口）

## 這份筆記在 MOC 扮演什麼角色
這份筆記的定位是「線性入口」：把 Mamba-1/2/3 的動機、代價與銜接關係講清楚，然後把推導細節拆到原子卡片。

若你想用最順的方式讀一遍，請從：[[Mamba/Mamba - 00 演化索引（線性脈絡）]] 開始。

---

## 建議閱讀順序
1. 核心矛盾（能力 × 硬體）：
	-  [[Mamba/Mamba - 01 核心矛盾：SSM 能力 × 硬體特性]]
2. 起點：為什麼 S4 在 NLP 不夠（內容感知）：
	-  [[Mamba/Mamba - 02 起點：為什麼 S4 不夠？（內容感知）]]
3. Mamba-1：引入 Selectivity（並承擔失去 FFT 的代價）：
	- Selective SSM 核心公式：[[Mamba/Mamba - 03 Mamba-1：Selective SSM（B_t,C_t,Δ_t）]]
	- $\Delta_t$ 作為動態門控直覺：[[Mamba/Mamba - 04 Mamba-1：Δ_t 作為資訊過濾與動態門控]]
	- 工程補救：Parallel Scan：[[Mamba/Mamba - 05 Mamba-1：為何失去 FFT？以及 Parallel Scan 的位置]]
4.  Mamba-2：SSD（把計算拉回 GEMM，提升 Tensor Core 吃滿）：
	- 問題設定與限制：[[Mamba/Mamba - 06 Mamba-2：SSD 的問題設定與限制]]
	- 矩陣形式與 chunk 策略：[[Mamba/Mamba - 07 Mamba-2：SSD 的矩陣形式與 Block-Chunk 策略]]
5. Mamba-3：Inference-oriented（以推論 memory-bound 為前提，換取更強 state tracking）：
	- 指數梯形離散化：[[Mamba/Mamba - 08 Mamba-3：Exponential-Trapezoidal Discretization]]
	- 相位/週期追蹤（RoPE 直覺）：[[Mamba/Mamba - 09 Mamba-3：Complex-valued Dynamics via RoPE（直覺）]]
	- MIMO / rank-$R$（提高 arithmetic intensity）：[[Mamba/Mamba - 10 Mamba-3：MIMO 與 memory-bound 觀點]]
6. 模塊拓樸的「移除與融合」總結：
	- 原子卡：[[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]
	- 入口筆記：[[Mamba 架構的拓樸演化：從 H3 到極致精簡的 Mamba-3]]

---

## 一頁總覽：每一代做了什麼交換

| 世代 | 想解的核心矛盾 | 核心做法 | 主要代價 |
|---|---|---|---|
| S4 | 高效長序列 | LTI SSM + 卷積化（FFT） | 缺乏內容感知 |
| Mamba-1 | 內容選擇性 | data-dependent dynamics（Selective SSM） | 失去固定卷積核（FFT 不再適用） |
| Mamba-2 | GPU 訓練吃滿 | SSD：把大量計算轉回矩陣乘法 | transition 結構限制更強，dynamics 自由度下降 |
| Mamba-3 | 推論品質/追蹤能力 | 以 memory-bound 角度增加 dynamics 計算 | dynamics 更複雜、需要更穩定的架構設計 |

---

# Mamba-1：讓 SSM 具備內容感知能力

## 重點一句話
把「內容依賴」塞進 SSM 的狀態更新：重要 token 更新得快、次要 token 幾乎不動。

## 最小公式集（抓住 Selective SSM 的本質）
Mamba-1 讓下列量變成 input-dependent：

$$
B_t = \text{Linear}_B(x_t),\quad
C_t = \text{Linear}_C(x_t),\quad
\Delta_t = \text{Softplus}(\text{Parameter}+\text{Linear}_{\Delta}(x_t))
$$

並透過離散化把動態性灌進 transition：

$$
\bar A_t = \exp(\Delta_t A)
$$

這讓 $\Delta_t$ 扮演類 gate 的角色（刷新 vs 保留）。

深讀：[[Mamba/Mamba - 03 Mamba-1：Selective SSM（B_t,C_t,Δ_t）]]、[[Mamba/Mamba - 04 Mamba-1：Δ_t 作為資訊過濾與動態門控]]

## 代價與工程補救
代價是失去固定卷積核，因此 FFT 不再直接適用；補救是把 recurrence 改寫成可 parallel scan。

深讀：[[Mamba/Mamba - 05 Mamba-1：為何失去 FFT？以及 Parallel Scan 的位置]]

---

# Mamba-2：Structured State Space Duality (SSD)

## 重點一句話
承認 GPU 的強項是 GEMM，SSD 把大量計算轉回矩陣乘法，換取更高吞吐。

## SSD 的核心視角
SSD 將整段序列計算寫成矩陣乘法，並指出 SSM 與 linear attention 在某些結構限制下存在「可互換的矩陣描述」。

深讀：[[Mamba/Mamba - 06 Mamba-2：SSD 的問題設定與限制]]、[[Mamba/Mamba - 07 Mamba-2：SSD 的矩陣形式與 Block-Chunk 策略]]

---

# Mamba-3：Inference-oriented SSM

## 重點一句話
把推論視為 memory-bound：在不顯著增加 latency 的前提下，用更多 dynamics 計算換更強 state tracking。

## 三個改造（對應三種補強）
- 離散化更精確：[[Mamba/Mamba - 08 Mamba-3：Exponential-Trapezoidal Discretization]]
- 週期/相位追蹤：[[Mamba/Mamba - 09 Mamba-3：Complex-valued Dynamics via RoPE（直覺）]]
- 狀態交互容量提升：[[Mamba/Mamba - 10 Mamba-3：MIMO 與 memory-bound 觀點]]

另外，從模塊拓樸角度看「移除與融合」：[[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]