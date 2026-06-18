# Failure Modes

這份文件不講正常流程，而是整理最常壞掉的地方。

## 1. 資料找不到

### 症狀

`collect_input_files()` 找不到任何輸入檔案。

### 原因

- `data_dir` 指錯。
- 資料是資料夾模式，但 `data_folder_count` 或 `data_file_name` 不對。
- 資料是 flat 模式，但副檔名與搜尋模式不吻合。

### 對策

先確認資料布局，再決定要走資料夾模式還是傳統 glob 模式。

## 2. benign 為空

### 症狀

`stage1_parse()` 或 `stage2_bert()` 直接失敗。

### 原因

- label 規則把 benign 誤判掉。
- 原始資料沒有 benign 事件。

### 對策

先檢查 `ParsedEvent.label` 與 `label_map` 的一致性。

## 3. label_mode 與 cache 不一致

### 症狀

入口檔在讀 cache 時直接報錯。

### 原因

- 之前是 `none`，這次改成 `binary` 或 `technique`。
- 舊 cache 沒有對應新的 label_map。

### 對策

從 `start_step=1` 重新跑，讓 stage 1 重建 label map 與 meta。

## 4. window 跨檔案邊界

### 症狀

模型看到了不合理的上下文，訓練表現和推論表現都變差。

### 原因

- `file_segments` 不對。
- dataset builder 失去 segment 邊界資訊。

### 對策

確認 stage 1 的 segment 計數與 stage 4 的輸入一致。

## 5. template / para 維度對不上

### 症狀

`BiLSTMAttention` 或 loss 計算報 shape error。

### 原因

- `window_size` 改了。
- `resource_dim` 改了。
- `template_dim` 與 `para_dim` 來自不同版本的向量庫。

### 對策

重建 stage 2~4，讓 feature space 從同一批設定重新產生。

## 6. score leakage

### 症狀

異常分數過於樂觀，eval 表現看起來不合理地好。

### 原因

- 用到 eval data 估計 score scale。
- benign / eval 的分段界線混掉。

### 對策

只用 benign 訓練樣本估計 score scale，並保留 segment 邊界。

## 相關

- [[TPLogAD-01-offline-trainer-entrypoint]]
- [[TPLogAD-07-dataset-and-memmap]]
- [[TPLogAD-08-cache-and-rerun]]