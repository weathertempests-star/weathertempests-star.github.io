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
**SSM 已經能卷積化/FFT 加速了，為什麼還需要 S4？S4 解決哪個計算瓶頸？**

它把「長序列理論（HiPPO）」「可平行訓練（卷積/FFT）」與「可計算的結構化參數」串起來。

---

## S4 (Structured State Space) 模型與運算優化

### S4 與 HiPPO 的衝突與解決

- **問題**：計算卷積核需要算矩陣冪次（例如 $A^L$）。如果 $A$ 是一個像 HiPPO 那樣稠密矩陣，計算量會爆炸。

- **解法**：將 $A$ 變換為**對角矩陣 (Diagonal Matrix)**。
  - 對角矩陣只有對角線上有數字，其餘皆為 0
  - **運算優勢**：計算對角矩陣的 $L$ 次方，只需將對角線上的每個數字各自算 $L$ 次方即可（純量運算）

---

## 下一步
- 想回到卷積核 $K$ 的定義與推導：[[SSM/SSM - 08 卷積核 K 的構造與 FFT 複雜度]]
- 想把「SSM」接回「Mamba 的 selection / input-dependent 參數」：回到 [[Mamba 架構演化脈絡]]
