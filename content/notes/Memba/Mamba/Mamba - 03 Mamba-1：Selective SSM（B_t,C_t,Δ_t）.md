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
這張卡回答：**Mamba-1 到底「動態化」了什麼？為什麼這等價於把內容選擇性引入 SSM？**

你可以把它當成從「LTI SSM」走向「input-dependent dynamics」的最小公式集。

---

# Mamba-1：讓 SSM 具備內容感知能力

## Mamba-1 的核心目標
Mamba-1 的主要任務：
> 打破傳統 SSM 的 LTI 限制。

其核心思想：
- 不再讓所有 token 共用固定動力系統
- 改為根據當前輸入動態調整狀態更新方式

也就是：
> 將 SSM 從「靜態系統」轉變為「輸入依賴系統」。

---

## Selective SSM：Mamba-1 的核心突破

Mamba-1 引入：
- data-dependent parameters
- time-varying dynamics

讓模型能根據當前 token 動態調整狀態更新。

其核心形式：

- [[SSM/SSM - 03 參數矩陣 A-B-C-D 的幾何與物理解釋#^27b08f|輸入控制矩陣]]

$$
B_t = \text{Linear}_B(x_t)
$$

- [[SSM/SSM - 03 參數矩陣 A-B-C-D 的幾何與物理解釋#^3844bc|輸出觀察矩陣]]

$$
C_t = \text{Linear}_C(x_t)
$$

- [[SSM/SSM - 06 步長 Δ 的意義：時間解析度與動態門控#^6ab58c|離散化步長]]

$$
\Delta_t
=
\text{Softplus}
(
\text{Parameter}
+
\text{Linear}_{\Delta}(x_t)
)
$$

需要特別注意：Mamba-1 並不是直接令 $A_t = f(x_t)$。

真正動態化的是：

$$
\bar A_t = \exp(\Delta_t A)
$$

也就是：
- 原始連續時間矩陣 $A$ 仍是全域學習參數
- 但離散化後的 transition dynamics 會依賴輸入內容

因此：
> 不同 token 會對應不同的狀態更新速度與記憶衰減行為。

---

## 下一步
- $\Delta_t$ 作為資訊過濾的直覺：[[Mamba/Mamba - 04 Mamba-1：Δ_t 作為資訊過濾與動態門控]]
- 為何失去 FFT、又如何平行：[[Mamba/Mamba - 05 Mamba-1：為何失去 FFT？以及 Parallel Scan 的位置]]
