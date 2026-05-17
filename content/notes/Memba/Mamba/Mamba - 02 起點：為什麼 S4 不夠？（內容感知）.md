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
這張卡回答：**為什麼明明 S4 很有效率，卻仍難在 NLP 超越 Transformer？**

它把 Mamba 的「第一性動機」固定下來：不是更快而已，而是要把「內容選擇性」引入 SSM。

---

## 為什麼 S4 無法真正取代 Transformer？

雖然 [[SSM/SSM - 11 S4：Structured State Space 與對角化動機|S4]] 透過 LTI (Linear Time-Invariant) 特性與卷積化表示，大幅提升了長序列建模效率，並在音訊、訊號處理、圖像等領域取得良好成果，但其在自然語言建模上始終難以超越 Transformer。

根本原因在於：
> 自然語言需要「內容感知 (Content Awareness)」。

語言中的資訊重要性高度不均衡：
- 主詞、實體名稱、關鍵事件需要長期保留
- 停用詞、過渡詞、標點通常重要性較低

但傳統 S4 屬於 LTI 系統：
- 同一組狀態轉移動力學會套用到所有 token
- 序列卷積核在一次 forward pass 中固定不變
- 無法根據當前輸入內容動態調整資訊流

因此：
> S4 雖然能高效「記憶」，但無法高效「選擇該記什麼」。

Mamba 系列的核心突破，便是將「選擇性 (Selectivity)」正式引入 SSM。

---

## 下一步
- 直接進入 Mamba-1 的 Selective SSM：[[Mamba/Mamba - 03 Mamba-1：Selective SSM（B_t,C_t,Δ_t）]]
