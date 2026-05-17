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
**HiPPO 理論要落地成可線上更新的 SSM，需要什麼樣的 $A$ 結構？**

它是「理論→可用參數化」的最小配方，也解釋了為什麼後續會遇到稠密矩陣計算瓶頸。

---

## 初始化的方法：HiPPO-LegS

為了讓投影過程能透過線性微分方程 $x'(t) = Ax(t) + Bu(t)$ 進行在線（Online）更新，矩陣 $A$ 必須滿足特定的數學結構。

**HiPPO-LegS（Legendre translated by state）** 給出以下初始化形式：

對於 $n, k \in \{0, 1, \dots, N-1\}$：
- **下三角元素 ($n > k$)**：

$$A_{nk} = -(2n+1)^{1/2}(2k+1)^{1/2}$$

- **對角線元素 ($n = k$)**：

$$A_{nn} = -(n+1)$$

- **上三角元素 ($n < k$)**：

$$A_{nk} = 0$$

---

## 下一步
- 看這種結構的 $A$ 在計算卷積核時為何會變慢、以及 S4 的解法：[[SSM/SSM - 11 S4：Structured State Space 與對角化動機]]
