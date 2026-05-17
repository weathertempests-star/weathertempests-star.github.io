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
**為什麼 SSM 的「記憶力」主要取決於 $A$？$B,C,D$ 分別像什麼？**

它把控制理論參數翻譯成「可直接套用到序列模型直覺」的語言。

---

## 參數矩陣的幾何與物理意義

- 矩陣 $A \in \mathbb{R}^{N \times N}$ : _狀態轉移矩陣（State Transition Matrix）_:
  - 決定記憶本身如何隨時間演化
  - **即使沒有新的輸入**：系統內部狀態仍會自己變化，包含：
    - 記憶衰減（forgetting）
    - 週期振盪（oscillation）
    - 長期保留（long-term retention）
    - 穩定性（stability）

- $A$ 的 `Eigen Value` 意義：若 $A$ 的特徵值實部為大幅負值，記憶會快速衰減

|eigenvalue|行為|
|---|---|
|$\Re(\lambda) \ll 0$|快速衰減|
|$\Re(\lambda) \approx 0$|長期保留|
|$\Im(\lambda) \neq 0$|振盪|

> SSM 的長序列能力，本質上取決於如何設計 $A$。

---

- 矩陣 $B \in \mathbb{R}^{N \times 1}$ : _輸入控制矩陣_ ^27b08f
  - 決定外部輸入如何寫入記憶：將外部輸入訊號 $u(t)$ 投影至隱含狀態空間
  - Be like:
    - input encoder
    - information injector

- 矩陣 $C \in \mathbb{R}^{1 \times N}$ : _輸出觀察矩陣_ ^3844bc
  - 從內部記憶中讀取資訊：如何從高維 state 取出輸出訊號 $y(t)$
  - 為何需要：state 可能是 high dimension，但輸出通常只需要部分資訊
  - Be like:
    - read head
    - observation layer

- 矩陣 $D \in \mathbb{R}^{1 \times 1}$ : _前饋矩陣（Feedthrough Matrix）_
  - 代表輸入對輸出的直接影響，不經過 state
  - Be like:
    - residual connection
    - shortcut path
  - 在深度學習 SSM 中，常被省略或簡化成 residual branch

> 此一連續微分架構確保了理論上可以處理無限長度的連續訊號。

---

## 下一步
- 進入離散化：[[SSM/SSM - 04 離散化：為何必要與 ZOH 假設]]
