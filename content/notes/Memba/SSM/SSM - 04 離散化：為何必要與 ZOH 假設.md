---
type: atomic
tags:
  - Memba
  - model-structure
  - SSM
status: active
created: 2026-05-15
updated: 2026-05-15
authors: WeiTa
---

## 這張卡在整體脈絡扮演什麼角色
**為什麼一定要把連續時間 SSM 變成離散遞迴？ZOH 是什麼假設？**

它把「控制理論的連續 ODE」接到「可在數位硬體上跑的序列模型」的橋樑搭起來。

---

## 離散化 (Discretization) 與 ZOH 機制

### 單純的 SSM 是不夠的
SSM 的原始定義基於**連續時間系統**的線性常微分方程 (ODE)：`（此處 $Du(t)$ 省略）`

$$
\begin{gather}
 x'(t) = Ax(t) + Bu(t) \\
 y(t) = Cx(t)
\end{gather}
$$

但是數位硬體僅能處理離散序列 $u_0, u_1, \dots$。
所以需要**離散化**：將連續算子轉換為離散遞迴算子，使模型能在電腦上進行運算。

### Zero-Order Hold (ZOH)
> ZOH 假設在時間步長 $\Delta$ 內，輸入訊號 $u(t)$ 保持不變。
> i.e.  $u(t) = u_k$ | $t \in [t_k, t_k + \Delta]$。

---
	
## 下一步
- 看完整 ZOH 推導：[[SSM/SSM - 05 ZOH 離散化推導：Abar, Bbar]]
- 先看 $\Delta$ 到底在控制什麼：[[SSM/SSM - 06 步長 Δ 的意義：時間解析度與動態門控]]
