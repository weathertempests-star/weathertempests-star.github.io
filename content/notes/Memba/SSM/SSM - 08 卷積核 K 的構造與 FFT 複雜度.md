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
**LTI SSM 的輸出為什麼等價於離散卷積？卷積核 $K$ 怎麼寫？FFT 加速的複雜度怎麼來？**

這張卡是把「SSM 訓練可平行化」落到可計算的數學形式。

---

## 用 CNN 的角度訓練 SSM：將「時間」轉換為「公式」
- LTI 特性與卷積：
  - 因為規則（系統參數 $(A, B, C)$）不變，第 $k$ 個輸出的結果其實是前面所有輸入的「加權總和」
  - 因此，可以預先計算出一個長度為 $L$ 的**影響力模板**（Convolution Kernel $K$）
  - **訓練時**：直接將模板套在整段輸入序列上（像 CNN 一樣掃過去），所有位置的輸出一次算完

- **卷積核構造**：

$$K = (C\bar{B},\; C\bar{A}\bar{B},\; C\bar{A}^2\bar{B},\; \dots,\; C\bar{A}^{L-1}\bar{B})$$

- 可以用 **FFT**：
  - 透過卷積定理，時域卷積等於頻域乘法
  - 使用 FFT（快速傅立葉轉換），複雜度從 $O(L^2)$ 降至 $O(L \log L)$

---

## S4 的數學解釋（卷積等價性的推導）

### 1) SSM 的離散形式定義
1. **狀態更新**：$x_k = \bar{A}x_{k-1} + \bar{B}u_k$
2. **輸出方程式**：$y_k = Cx_k$

假設初始狀態 $x_{-1} = 0$，將時間步 $k$ 逐步展開：

- $k=0$：$x_0 = \bar{B}u_0$
- $k=1$：$x_1 = \bar{A}(\bar{B}u_0) + \bar{B}u_1$
- $k=2$：$x_2 = \bar{A}^2\bar{B}u_0 + \bar{A}\bar{B}u_1 + \bar{B}u_2$

以此類推：

$$x_k = \sum_{i=0}^k \bar{A}^{k-i} \bar{B} u_i$$

### 2) 映射至輸出空間

代入 $y_k = Cx_k$：

$$y_k = \sum_{i=0}^k (C \bar{A}^{k-i} \bar{B}) u_i$$

定義：

$$K_j = C \bar{A}^j \bar{B}$$

其中 $j$ 代表距離當前時間點的步數差（Lag）。

則：

$$y_k = \sum_{i=0}^k K_{k-i} u_i$$

這等價於離散卷積：

$$y_k = (K * u)_k = \sum_{i=0}^k K_i u_{k-i}$$

---

## 下一步
- 看為何 HiPPO 會讓 $A$ 變稠密、導致計算卷積核很痛：[[SSM/SSM - 11 S4：Structured State Space 與對角化動機]]
