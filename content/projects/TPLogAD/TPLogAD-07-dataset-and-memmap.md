# Dataset and Memmap

這份文件描述 `DatasetBuilder` 的設計：把可重入事件流轉成 memmap-backed sliding window dataset，讓大型資料集不用先全部進 RAM。

## 為什麼要兩次 pass

- Pass 1：先探測第一筆事件，確認向量維度與總筆數。
- Pass 2：再把整個事件流寫進 memmap。

這樣設計的原因很直接：資料量大時，不能先把所有事件展開成 list。

## Sliding window 的語意

每個樣本不是單筆事件，而是一個 window：

- `X_window`：`window_size × input_dim`
- `y_tmpl`：window 結尾那筆事件的 template class
- `y_para`：window 結尾那筆事件的參數向量
- `y_lbl`：若啟用 label supervision，還會帶一個 label class

## 為什麼不能跨檔案切窗

不同輸入檔代表不同事件序列，若 window 穿越檔案邊界，模型會看到不合理的上下文。`file_segments` 就是用來保護這個邊界的。

## memmap 的好處

- 樣本可以直接從磁碟映射讀取。
- 不需要把整個 dataset 複製到 RAM。
- `__getitem__` 只拼出當前 window，不會持有整批資料。

## 相關

- [[TPLogAD-02-pipeline-contracts]]
- [[TPLogAD-05-para2vec-and-resource-encoding]]
- [[TPLogAD-08-cache-and-rerun]]