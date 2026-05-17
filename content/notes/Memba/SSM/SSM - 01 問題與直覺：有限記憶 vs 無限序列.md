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
**為什麼需要 SSM？

在 Mamba-SSM 的脈絡中，SSM 提供一種「不回頭看全部 token」的序列建模方式：
- 透過持續更新的內部狀態 $x(t)$ 壓縮歷史
- 以 $O(1)$ 記憶體（推論）承接長序列


---

## SSM 是甚麼
狀態空間模型（State Space Model, SSM）本質上是在解決：
> 「如何用有限大小的記憶體，持續理解無限長序列？」

可以將它想成一個「持續更新內部記憶的系統」。

例如閱讀句子時：

> “The cat that chased the mouse was hungry.”

模型看到 `was hungry` 時，必須還記得主詞是 `cat`

Transformer 的做法是：
- 回頭重新查看前面所有 token（Attention）( [[Transformer 的問題是甚麼]])

SSM 的做法則是：
- 不回頭看
- 而是把過去資訊壓縮進一個「狀態（state）」裡持續更新

### 和其他模型有甚麼不同

| 模型          | 如何記住過去          |
| ----------- | --------------- |
| RNN         | 小型 hidden state |
| Transformer | 保存全部 token      |
| SSM         | 用動態系統壓縮歷史       |

SSM 的核心思想：
> 「把序列建模問題，轉換成物理系統隨時間演化的問題。」

---

## 下一步
- 想先抓住數學骨架：[[SSM/SSM - 02 連續時間 LTI SSM 定義 (A,B,C,D)]]
- 想先理解為何能上 GPU 平行訓練：[[SSM/SSM - 07 遞迴視角 vs 卷積視角：為何可平行訓練]]
