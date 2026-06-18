# Pipeline Contracts

詳細紀錄每個 stage 期待什麼輸入、產生什麼輸出、哪些欄位不能亂改。

## 核心物件

### `PipelineConfig`

`PipelineConfig` 是整條 pipeline 的單一設定來源。它把下列資訊集中在一起：

- 資料布局：`data_dir`、`data_folder_count`、`data_file_name`、`input_recursive`
- 特徵工程：`normalizer`、`use_ants`、`kmeans_k`、`resource_dim`
- 序列模型：`window_size`、`lstm_hidden`、`lstm_layers`、`dropout`
- 訓練控制：`epochs`、`batch_size`、`lr`、`require_cuda`
- 流程控制：`start_step`、`force_reparse`、`force_retrain`
- 監督式分類：`label_mode`、`label_loss_weight`

### `CachePaths`

`CachePaths` 的角色是把快取路徑標準化，避免每個模組自己拼字串。它把 stage 輸出固定在 `weights/` 下，讓 rerun 與 debug 都有一致的視角。

### `ParsedMeta`

`ParsedMeta` 是 stage 1 留給後面 stage 的摘要資料。它保存：

- `n_templates`：模板類別數
- `train_segments`：benign 事件分段計數
- `eval_segments`：全部事件分段計數
- `label_mode`：當次訓練使用的標籤模式
- `label_map`：label 字串到 class id 的對照表

## 為什麼這些物件重要

- `stage4_dataset` 需要 segment 計數，否則 window 可能跨檔案邊界。
- `stage5_train` 需要 `n_templates` 與 `para_dim`，否則輸入/輸出形狀對不上。
- `label_map` 一旦變了，舊 cache 不能再直接沿用。


## 相關

- [[TPLogAD-01-offline-trainer-entrypoint]]
- [[TPLogAD-03-parsing-and-labels]]
- [[TPLogAD-07-dataset-and-memmap]]
- [[TPLogAD-08-cache-and-rerun]]