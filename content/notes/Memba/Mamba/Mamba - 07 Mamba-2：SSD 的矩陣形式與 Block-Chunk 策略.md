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
這張卡回答：**SSD 具體把整段序列重寫成什麼矩陣形式？以及它如何用 chunk 把「Tensor Core 優勢」與「SSM 的 state 傳遞」同時保留？**

---

## SSD 的矩陣形式

SSD 將整段序列的計算寫成：

$$
y = Mx
$$

其中：

$$
M
=
L
\circ
(CB^\top)
$$

這裡：

| 符號 | 意義 |
|---|---|
| $CB^\top$ | token 之間的 pairwise interaction |
| $L$ | 因果衰減結構 |
| $\circ$ | Hadamard product |

---

## SSD 與線性注意力的對偶性

SSD 的重要發現：
> 某些 SSM 可被解讀為線性 attention。

若：
- 將部分 transition dynamics 特殊化
- 並重新排列計算形式

則 SSM 可映射到類似 $QK^\top V$ 的 attention 結構。

注意：這不是宣稱與 Transformer 等價；而是指出共享某種底層數學結構。

---

## Block Matrix Decomposition（區塊化混合策略）

### 區塊內 (Intra-chunk)
長序列先被切成 chunk。
在 chunk 內：
- 使用 attention-like matrix computation
- 利用 Tensor Core 執行高密度 GEMM

這部分具有：
- 高 arithmetic intensity
- 高 GPU utilization

### 區塊間 (Inter-chunk)
chunk 與 chunk 之間：
- 仍使用 SSM recurrence 傳遞 compressed state

因此：
- 保留線性時間特性
- 保留 constant-memory decoding

---

## 下一步
- 若想理解為何 Mamba-3 會轉向推論 memory-bound 視角：[[Mamba/Mamba - 08 Mamba-3：Exponential-Trapezoidal Discretization]]
