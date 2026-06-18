# Cache and Rerun

這份文件專門說明快取與重跑策略。
TPLogAD 的 pipeline 本質上是可重入的，所以快取不是附屬功能，而是設計的一部分。

## 快取怎麼命名

`CachePaths` 會把所有主要產物固定到 `weights/` 下，名稱中包含 `ants/raw`、`window_size`、`resource_dim`、`kmeans_k` 等資訊，避免不同設定互相覆蓋。

## `start_step` 的意義

`start_step` 允許你從中間 stage 開始重跑，但前提是前置快取仍然有效。

- `1`：重新解析資料。
- `2`：重新適配文字模型。
- `3`：重新建立向量元件。
- `4`：重新建 dataset。
- `5`：只重訓模型。

## `force_reparse` 與 `force_retrain`

- `force_reparse`：強制重做 stage 1。
- `force_retrain`：強制重做可重建的模型與向量產物。

這兩個旗標的差異很重要，因為 stage 1 與後面 stage 的 cache 生命週期並不相同。

## 什麼時候不能直接重用 cache 

**! 很重要 !**

- label_mode 改了。
- 資料路徑結構改了。
- `window_size` / `resource_dim` / `kmeans_k` 改了。
- ANTs 與 raw 互換了。

## 相關

- [[TPLogAD-02-pipeline-contracts]]
- [[TPLogAD-01-offline-trainer-entrypoint]]
- [[TPLogAD-09-failure-modes]]