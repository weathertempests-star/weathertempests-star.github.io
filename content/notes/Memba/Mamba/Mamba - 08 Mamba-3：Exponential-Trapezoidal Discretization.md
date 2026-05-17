---
type: atomic
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

## 這張卡在整體脈絡扮演什麼角色
這張卡回答：**Mamba-3 為什麼要改離散化？改了之後 state update 形式多了什麼、解了什麼問題？**

它也提供一個重要 bridge：為什麼「移除顯式卷積」在拓樸上是合理的。

---

# Mamba-3：Inference-oriented SSM

## Mamba-3 的核心方向

Mamba-3 的思路轉向：
> 推論階段其實通常是 Memory-bound。

也就是 GPU 很多時候不是算不動，而是在等待資料搬運。

因此 Mamba-3 嘗試利用閒置 ALU：
- 執行更複雜 dynamics
- 提升 state tracking 能力
- 在低額外 latency 下提升模型能力

---

## 技術 1：Exponential-Trapezoidal Discretization

### 梯形積分觀點
Mamba-3 引入 trapezoidal-style discretization。
核心思想：
- 過去方法主要依賴單端點資訊
- 梯形法則同時考慮區間起點與終點

因此對連續 dynamics 的近似更準確。

### 更新形式

$$
h_t
=
\alpha_t h_{t-1}
+
\beta_t B_{t-1}x_{t-1}
+
\gamma_t B_tx_t
$$

重要意義：
> 當前 state 不再只依賴當前 token，而是同時顯式吸收前一步輸入。

### 直覺理解
從訊號處理角度：
> 這近似於在 recurrence 前加入短距離 data-dependent convolution。

效果：
- 短期特徵更平滑
- local transition 更穩定
- dynamics approximation 更精確

---

## 下一步
- 這一步如何對應到「隱含卷積」與移除顯式卷積：[[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]
- 週期/相位追蹤能力：[[Mamba/Mamba - 09 Mamba-3：Complex-valued Dynamics via RoPE（直覺）]]
