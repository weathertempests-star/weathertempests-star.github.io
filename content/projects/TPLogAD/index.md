# TPLogAD MOC

TPLogAD 的核心不是單一模型，而是一個從原始 SAGA 事件到異常分數與報表的離線 pipeline。

## 專案在做什麼

TPLogAD 以 SAGA JSONL 為輸入，經過 Drain3 解析、ANTs/原始字串標準化、template embedding、para2vec、BiLSTM-Attention 訓練，最後輸出 checkpoint、history、predictions 與vizualization result。
我們關心的是系統呼叫序列中的正常模式，並把偏離正常模式的事件視為 anomaly signal。

## 先讀的卡片

- [[TPLogAD-00-project-overview]]：專案全景與資料流心智模型。
- [[TPLogAD-01-offline-trainer-entrypoint]]：`pipeline/offline_trainer.py` 如何把整條流程串起來。
- [[TPLogAD-02-pipeline-contracts]]：`PipelineConfig`、`CachePaths`、`ParsedMeta` 的契約。
- [[TPLogAD-03-parsing-and-labels]]：SAGA 事件如何轉成 `ParsedEvent`，以及 label 如何進入 pipeline。
- [[TPLogAD-04-template-embedding]]：`itemplate2vec` 與 SecureBERT 適配。
- [[TPLogAD-05-para2vec-and-resource-encoding]]：參數向量與 `resource text` 的組合方式。
- [[TPLogAD-06-model-and-loss]]：BiLSTM-Attention 與雙任務/監督式 loss。
- [[TPLogAD-07-dataset-and-memmap]]：memmap dataset 與 sliding window 邊界控制。
- [[TPLogAD-08-cache-and-rerun]]：快取命名、重跑策略與 `start_step`。
- [[TPLogAD-09-failure-modes]]：最常壞掉的地方與 debug 順序。

## 使用方式

### 1. 資料放置

預設支援兩種資料布局：

- 資料夾模式：`/***/<num of dataset>/synthesized_events_0.json` 這類結構。
- 傳統模式：`data/raw/*.json` 或 `*.jsonl`。

### 2. 直接跑整條 pipeline

```bash
python pipeline/offline_trainer.py --data_dir /campaign_1hour --data_folder_count 5 --data_file_name synthesized_events_0.json --use_ants --label_mode none
```

### 3. 常用

- `--start_step`: 從哪個 stage 重新開始。
- `--force_reparse`: 強制重做 stage 1。
- `--force_retrain`: 強制重做 stage 2~5 的可重建產物。
- `--use_ants / --no_use_ants`: 切換語意標準化。
- `--label_mode none|binary|technique`: 控制是否加上監督式分類任務。

### 4. 產出位置

- `weights/securebert_v2/`：適配後文字模型。
- `weights/embeddings/`：template vocabulary 與 template vectors。
- `weights/para_encoder.pkl`：para2vec encoder。
- `weights/bilstm_best.pt`：最佳 checkpoint。
- `weights/model_config.pkl`：模型形狀與訓練契約。
- `weights/training_history.csv`：訓練歷史。
- `weights/predictions.csv`：推論分數與標註結果。


## 相關

- [[TPLogAD-00-project-overview]]
- [[TPLogAD-01-offline-trainer-entrypoint]]
- [[TPLogAD-02-pipeline-contracts]]