# ParaVec and Resource Encoding

這張卡片整理參數向量化與 resource text 的生成方式。這一層的目標不是完整重建原始事件，而是把事件中最有語意的可變欄位壓成固定長度向量。

## 五類參數編碼器

- `TimeEncoder`：把 timestamp 轉成週期性 sin/cos 特徵。
- `UserIDEncoder`：把 process/user identity 轉成 hash-based 連續值。
- `NumericalEncoder`：對數值型參數做 Z-score。
- `StateEncoder`：把離散 relation 壓成穩定的 state scalar。
- `ResourceEncoder`：把路徑、Registry key、URL 或 token 字串轉成 TF-IDF + SVD 表示。

## `ParaVecEncoder` 的角色

`ParaVecEncoder` 是統一入口。它的責任是：

- 只用訓練集資料做 fit，避免 leakage。
- 將一筆事件的多種參數壓成固定維度向量。
- 保證輸出長度固定，這樣才能和 template vector 在時間維度上拼接。

## `ParamKMeansSelector` 在這裡做什麼

很多參數字串太碎，不適合直接送進 resource encoder，所以會先用輕量特徵做 K-means，挑出代表性字串，再拿代表字串去構造 resource text。

這個步驟的重點不是準確分群，而是把大雜訊壓成少數可學習的代表型態。

## `resource text` 是什麼

`_build_resource_text()` 會把 destination 名稱與少量代表性參數組成一個壓縮字串。這個字串同時影響 template embedding 與 para2vec，所以它是兩條表示路徑的交會點。

## 相關

- [[TPLogAD-04-template-embedding]]
- [[TPLogAD-07-dataset-and-memmap]]
- [[TPLogAD-09-failure-modes]]