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
這張卡把 Mamba-1 的關鍵參數 $\Delta_t$ 翻譯成「可操作的直覺」：你可以把它當成一種 soft gate，控制刷新 vs 保留。

---

## $\Delta_t$ 的資訊過濾機制

Mamba 中最關鍵的動態參數其實是：

$$
\Delta_t
$$

它控制：
- 狀態衰減速度
- 記憶保留程度
- 新資訊寫入強度

其功能上類似於：
- RNN/LSTM 的 gating 行為

但並非真正的 gate 結構。

---

### 情況 1：重要資訊
例如：
- 人名
- 關鍵事件
- 主題詞

模型傾向預測較大的：

$$
\Delta_t
$$

此時：

$$
\bar A_t = \exp(\Delta_t A)
$$

會產生更強的歷史衰減效果。

結果：
- 舊狀態快速淡化
- 新資訊大量寫入

直覺上相當於：
> 「刷新記憶」。

---

### 情況 2：低資訊量 token
例如：
- the
- of
- 標點
- 過渡詞

模型傾向預測較小的：

$$
\Delta_t
$$

此時狀態變化非常小。

結果：
- 歷史狀態幾乎完整保留
- 當前 token 幾乎不影響記憶

直覺上相當於：
> 「忽略這個 token」。

---

## 下一步
- 為何 data-dependent dynamics 會讓 FFT 失效：[[Mamba/Mamba - 05 Mamba-1：為何失去 FFT？以及 Parallel Scan 的位置]]
