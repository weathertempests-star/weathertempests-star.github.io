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
這張卡回答：**Mamba-3 為什麼引入 MIMO / rank-$R$？這件事如何對應到「推論是 memory-bound」的硬體現實？**

---

## 技術 3：MIMO (Multi-Input Multi-Output)

### Mamba-1/2 的限制
早期 Mamba 本質較接近：SISO (Single-Input Single-Output)
- 每個 timestep 的 state update 較偏向向量級更新

這限制了：
- 多頭狀態混合能力
- dynamics interaction capacity

---

### MIMO 的核心思想
Mamba-3 引入：
- rank-$R$ structure
- matrix-level interaction

讓狀態更新從：
- vector operation

升級為：
- matrix-style interaction

---

### 為什麼這在推論時特別重要？
因為 autoregressive decoding 通常是 memory-bound。
GPU 真正瓶頸不是 FLOPs，而是：
- memory bandwidth
- cache miss
- weight loading

因此很多 ALU 其實是閒置的。

---

### MIMO 的核心價值
MIMO 提升 arithmetic intensity：
> 每次 memory access 能完成更多運算。

因此在 memory stall 期間，GPU 可以額外完成更複雜 dynamics computation。

結果：
- 更強 state aggregation
- 更高 representation capacity
- 極低額外 latency 成本

---

## 下一步
- 從模塊層面看 Mamba-3 還做了哪些「融合/移除」：[[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]
