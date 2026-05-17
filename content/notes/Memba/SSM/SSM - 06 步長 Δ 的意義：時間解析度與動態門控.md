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
**$\Delta$ 不是只是採樣間隔；它在深度學習 SSM 裡到底控制了什麼？**

在 Mamba 裡，$\Delta$ 會變成 input-dependent 且可學習的「動態門控」，直接影響遺忘/保留。

---

## 步長 $\Delta$ 的物理意義：時間解析度與過濾器

^6ab58c

$\Delta$ 不僅是採樣間隔，更是模型感知時間尺度的關鍵**參數**：

- **時間解析度 (Time Resolution)**：
  - **較大的 $\Delta$**：採樣間隔寬，模型「看」得更遠，傾向於捕捉宏觀、低頻的長期趨勢（Long-term context），自動過濾掉高頻噪聲。
  - **較小的 $\Delta$**：採樣間隔窄，模型「看」得更細，專注於捕捉微觀、高頻的瞬態細節（Short-term details）。

- **動態門控 (Dynamic Gating)**：
  在 Mamba 等現代 SSM 中，$\Delta$ 是輸入相關的（Input-dependent）且可學習的。
  這意味著模型能根據當前 token 的重要性，自主決定要「遺忘」多少舊狀態（透過較大的 $\Delta$ 增加衰減）或「保留」多少資訊。

---

## 下一步
- 讀離散遞迴形式與複雜度：[[SSM/SSM - 07 遞迴視角 vs 卷積視角：為何可平行訓練]]
