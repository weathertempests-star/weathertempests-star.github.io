# Project Overview

TPLogAD 的問題設定可以簡單用一句話理解：把大量系統事件轉成可學習的序列表示，讓模型從 benign 行為中學出正常模式，並在推論時對偏離正常模式的片段提高分數。

## 心智模型

這個專案不是單純的分類器，也不是單純的序列模型。它把一筆事件拆成兩條資訊流：

- `template`：事件的固定結構，描述「做了什麼」。
- `parameters`：事件中的可變欄位，描述「對誰、在哪裡、用什麼值做」。

前者主要靠 `itemplate2vec` 表示，後者主要靠 `para2vec` 表示。最後兩者在時間維度上拼接成一個 sliding window，再交給 `BiLSTMAttention` 做序列建模。

## 資料到分數

1. 讀取 SAGA JSONL。
2. 用 Drain3 抽出 template 與 parameters。
3. 視需要套用 ANTs 標準化。
4. 產生 template vector 與 para vector。
5. 建立 memmap dataset，避免把整批資料放進 RAM。
6. 訓練 BiLSTM-Attention。
7. 匯出 checkpoint、history、predictions 與 score scale。

## 限制

- 訓練資料只用 benign 序列，否則模型會把 anomaly 當成正常模式學進去。
- `template_dim + para_dim` 必須和模型輸入完全對齊。
- 若資料布局改成預設以外的資料夾結構，`collect_input_files` 的假設就會失效。
- 快取確立後改變訓練參數有可能會導致整體架構fail 

## 相關

- [[TPLogAD-01-offline-trainer-entrypoint]]
- [[TPLogAD-02-pipeline-contracts]]
- [[TPLogAD-03-parsing-and-labels]]
- [[TPLogAD-06-model-and-loss]]