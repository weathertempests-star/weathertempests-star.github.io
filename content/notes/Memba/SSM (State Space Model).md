---
type: permanent
tags:
  - Memba
  - model-structure
status: active
created: 2026-05-13
updated: 2026-05-15
authors: WeiTa
---
---


## 脈絡
### 1) 問題設定：有限記憶 vs 無限序列
SSM 想回答的是：**如何用有限大小的 state，持續承接無限長序列？**
深讀：[[SSM/SSM - 01 問題與直覺：有限記憶 vs 無限序列]]

### 2) 最小數學骨架：連續時間 LTI SSM
把「序列」看成動態系統的時間演化： ^31c58c

$$
\begin{aligned}
 x'(t) = Ax(t) + Bu(t) \\
 y(t) = Cx(t) + Du(t)
\end{aligned}
$$

深讀：
- 定義與符號：[[SSM/SSM - 02 連續時間 LTI SSM 定義 (A,B,C,D)]]
- $A,B,C,D$ 的角色直覺：[[SSM/SSM - 03 參數矩陣 A-B-C-D 的幾何與物理解釋]]

### 3) 連續 → 離散：讓模型能在數位硬體上跑
[[#^31c58c|連續 ODE]] 需要離散化成可遞迴更新的形式；ZOH 是最常見且可精確推導的假設。
深讀：
- 為何必要＋ZOH 假設：[[SSM/SSM - 04 離散化：為何必要與 ZOH 假設]]
- $\bar{A},\bar{B}$ 推導：[[SSM/SSM - 05 ZOH 離散化推導：Abar, Bbar]]
- $\Delta$ 的物理意義（含 Mamba gating 直覺）：[[SSM/SSM - 06 步長 Δ 的意義：時間解析度與動態門控]]

### 4) 訓練平行化：遞迴視角 vs 卷積視角
同一個離散 SSM：推論用遞迴（省記憶體）、訓練可轉成卷積（可平行）。
深讀：
- 視角切換：[[SSM/SSM - 07 遞迴視角 vs 卷積視角：為何可平行訓練]]
- 卷積核 $K$ 與 FFT 複雜度：[[SSM/SSM - 08 卷積核 K 的構造與 FFT 複雜度]]

### 5) 長記憶理論：HiPPO
HiPPO 把「有限 state 保留長期歷史」形式化成最優投影的壓縮協議。
深讀：
- 理論直覺：[[SSM/SSM - 09 HiPPO：長序列壓縮的最優投影]]
- LegS 初始化：[[SSM/SSM - 10 HiPPO-LegS 初始化：矩陣 A 的結構]]

### 6) 結構化與可計算性：S4
卷積化之後仍會遇到「稠密 $A$ 讓 $A^L$ 很難算」的瓶頸；S4 的核心是讓 $A$ 結構化（如對角化）以降低成本。
深讀：[[SSM/SSM - 11 S4：Structured State Space 與對角化動機]]

---

## 符號表
- $u(t)$：輸入訊號（可視為 token embedding 的某一維）
- $x(t)$：內部狀態／記憶（state）
- $y(t)$：輸出
- $A$：狀態轉移（記憶演化；長記憶能力多由此決定）
- $B$：輸入如何寫入 state
- $C$：如何從 state 讀出輸出
- $D$：輸入對輸出的直接路徑（深度學習版本常簡化成 residual）
- $\Delta$：離散化步長（也可被視為時間尺度/門控）
- $\bar{A}, \bar{B}$：離散化後的轉移/輸入矩陣
- $K$：卷積核（把遞迴視角轉成卷積視角時的影響力模板）