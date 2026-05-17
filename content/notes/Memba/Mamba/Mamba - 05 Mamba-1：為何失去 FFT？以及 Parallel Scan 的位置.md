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
這張卡回答兩件事：
1) 為什麼 Mamba-1 一引入 selectivity 就不能再用 S4 那套 FFT 卷積訓練？
2) 工程上它把 recurrence 改寫成什麼，才又能在 GPU 上平行？

---

## 為什麼 Mamba-1 失去 FFT 優勢？

S4 能高效訓練的重要原因：
> LTI 系統可被轉換為固定卷積。

因此：
- 可使用 FFT
- 可一次平行處理整段序列

但 Mamba-1 引入 data-dependent dynamics 後：
- 每個 timestep 的 dynamics 都不同
- 系統不再具有固定 convolution kernel

因此：
> 無法再直接使用 FFT convolution。

運算重新退化為遞迴形式。

---

## Hardware-aware Parallel Scan

雖然 recurrence 本身是序列性的，但：
> Mamba 將狀態更新重寫為具有結合律 (Associativity) 的 scan operation。

因此 GPU 可以使用 parallel prefix-scan 演算法進行平行化。

### 核心思想

傳統 recurrence：

$$
h_t = f(h_{t-1}, x_t)
$$

通常難以平行。

但若更新形式滿足結合律：

$$
(a \star b)\star c
=
a \star (b \star c)
$$

則可以進行 tree-style parallel scan。

---

### 硬體感知 (Hardware-aware)
Mamba 的 scan 設計高度依賴 GPU 記憶體階層：

| 記憶體 | 特性 |
|---|---|
| SRAM | 快、小 |
| HBM | 大、慢 |

核心目標：
> 盡量減少昂貴的 HBM 存取。

因此：
- 中間狀態盡可能保存在 SRAM
- 減少 memory traffic
- 提高 GPU utilization

---

## 下一步
- 若你想把「scan」當成 Mamba 的工程主角（而不只是附帶一段）：之後可以在 `## 3. Selective Scan 與 Hardware-Aware 設計` 再拆更細
- 下一代的方向：把計算拉回矩陣乘法：[[Mamba/Mamba - 06 Mamba-2：SSD 的問題設定與限制]]
