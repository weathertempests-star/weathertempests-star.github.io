# Parsing and Labels

這張卡片描述 stage 1 的核心：如何把 SAGA JSONL 轉成 `ParsedEvent`，以及 label 是怎麼進入 pipeline 的。

## 解析流程

1. 讀取 JSONL。
2. 從 `srcNode` / `dstNode` 抽出欄位。
3. 把事件序列化成給 Drain3 的單行文字。
4. 由 Drain3 抽出 `template` 與 `parameters`。
5. 補回結構化欄位，例如 UUID、PID、cmdline。
6. 若啟用 ANTs，就把路徑、Registry、Network 字串做標準化。

## `ParsedEvent` 的角色

`ParsedEvent` 不是原始 JSON 的簡單映射，而是後續特徵工程需要的中間表示。它同時保留：

- `template`
- `parameters`
- 原始欄位（src/dst/relation/timestamp/label）
- `cluster_id`

這讓後面的 template embedding、para2vec、dataset builder 都能用同一筆事件，不需要重新解析原始檔。

## label 的三種模式

### `none`

只做 anomaly detection，不額外訓練 label 分類頭。

### `binary`

把所有非 benign 標成 attack，適合先做最簡單的監督式輔助任務。

### `technique`

從 label 字串抽取 MITRE ATT&CK technique code，例如 `T1047`。如果抽不到，通常會退回 `unknown`。

## 容易壞的地方

- 原始資料沒有 benign，後續 stage 會直接失效。
- label 字串格式不一致，`label_map` 會和 cache 失配。
- Drain3 沒回傳有效 template，代表解析契約壞掉。
- ANTs 標準化不能成為單點失敗來源，所以任何例外都會退回原始字串。

## 相關

- [[TPLogAD-01-offline-trainer-entrypoint]]
- [[TPLogAD-04-template-embedding]]
- [[TPLogAD-05-para2vec-and-resource-encoding]]