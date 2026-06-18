# Offline Trainer Entry Point

`pipeline/offline_trainer.py` 是這個專案的主入口。
它不負責特徵工程，也不直接實作模型，而是把設定、快取與各個 stage 組裝起來保證整體流程閱讀性。

## What is for

- 讀取 `PipelineConfig`。
- 建立 `CachePaths`。
- 依 `start_step` 決定要重建哪些 stage。
- 把 benign/all 事件流以 factory 形式傳遞下去。
- 訓練完成後輸出 model config、history、predictions，必要時啟動可視化。

## Key components

### `start_step`

`start_step` 不是單純的 debug 參數，而是整條 pipeline 的重入控制開關。它決定：

- 要不要重做解析。
- 要不要重訓文字模型與向量元件。
- 要不要直接載入現有快取。

### `label_mode`

`label_mode` 會影響 dataset、loss 與輸出內容。

- `none`：只走原本的 anomaly detection 路徑。
- `binary`：把 label 壓成 benign / attack。
- `technique`：把 label 轉成技術類別，通常會從標籤字串中抽取 technique code。

### `use_ants`

`use_ants` 決定 normalizer 的選擇，直接影響 template 與 resource text 的語意粒度。

## why catch a benign event first?

入口檔在真正開始 stage 之前會先取一筆 benign event。
這不是多餘檢查，而是 cheapest sanity check：

- 可以早點發現資料是空的。
- 可以早點驗證 `template_dim` 與 `para_dim` 是否可推導。
- 可以避免後面 stage 深層才爆 `StopIteration`。

## 輸出

`run()` 完成後，至少會留下：

- `model_config.pkl`
- `training_history.csv`（若啟用）
- `predictions.csv`（若啟用）
- `bilstm_best.pt`

## 相關

- [[TPLogAD-02-pipeline-contracts]]
- [[TPLogAD-07-dataset-and-memmap]]
- [[TPLogAD-08-cache-and-rerun]]