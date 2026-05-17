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
這張卡回答：**為什麼 Mamba-3 會關心「複數/相位」？它想補足哪一類 state tracking 能力？**

---

## 技術 2：Complex-valued Dynamics via RoPE

### 為什麼實數 dynamics 不夠？
純實數 dynamics 對：
- 振幅衰減
- 平滑記憶

很有效。

但對某些需要：
- phase tracking
- periodic structure
- alternating state

的任務較不理想，例如：
- parity
- bracket matching
- periodic pattern

---

### 複數 dynamics 的優勢
複數狀態具有：

| 組件 | 功能 |
|---|---|
| Magnitude | 控制強度 |
| Phase | 控制旋轉與週期 |

相位旋轉天然適合：
- 狀態循環
- 週期追蹤
- phase memory

---

### 為什麼不直接使用 complex arithmetic？
直接 complex computation：
- kernel 實作困難
- GPU support 較差
- Tensor Core utilization 不理想

因此成本較高。

---

### Mamba-3 的做法
Mamba-3 利用 `RoPE` 將：
- complex phase rotation

轉寫為：
- real-valued rotation operation

也就是：
> 用實數空間中的幾何旋轉近似複數 phase dynamics。

效果：
- 提升週期性 state tracking 能力
- 避免昂貴 complex hardware cost

---

## 下一步
- 推論 memory-bound 的另一個改造：MIMO：[[Mamba/Mamba - 10 Mamba-3：MIMO 與 memory-bound 觀點]]
