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
這張卡回答：**Mamba-2 為什麼要做 SSD？它先承認了哪個硬體現實，並因此對 dynamics 加了哪些限制？**

---

# Mamba-2：Structured State Space Duality (SSD)

## Mamba-2 的核心問題

Mamba-1 雖然成功引入 Selectivity，但產生另一個問題：
> GPU Tensor Core 利用率不足。

原因在於：
- scan / recurrent algorithm 本質上仍偏序列化
- 不適合現代 GPU 最擅長的 dense matrix multiplication

因此：即使理論 FLOPs 很低，GPU 仍可能無法完全吃滿。

---

## Mamba-2 的核心思想

Mamba-2 試圖回答：
> 能否將 SSM 重新轉換回矩陣乘法形式？

這便導向 SSD。

SSD 的核心不是單純加速，而是：
> 建立 SSM 與線性注意力之間的數學對偶性。

---

## SSD 的關鍵限制

Mamba-2 對狀態轉移結構施加更強限制。
相較於 Mamba-1 的較自由 dynamics：

SSD 將 transition structure 特殊化為共享尺度形式。

此限制的核心目的：
> 讓 recurrence 可以被重新寫成矩陣運算。

---

## 下一步
- SSD 的矩陣形式與 chunk 混合策略：[[Mamba/Mamba - 07 Mamba-2：SSD 的矩陣形式與 Block-Chunk 策略]]
