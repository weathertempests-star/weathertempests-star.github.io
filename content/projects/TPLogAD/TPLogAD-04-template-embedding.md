# Template Embedding

這張卡片說明 itemplate2vec 在做什麼：把 Drain3 抽出的 template 轉成固定維度的向量，讓後面的序列模型能把它當成一般數值特徵。

## 核心想法

template 不是自然語言句子，而是高度結構化的 token 序列。若直接平均詞向量，太通用的 token 會蓋掉真正有辨識度的 token，所以這裡會先算 token 權重，再做加權 pooling。

## 流程

1. 收集所有 template token，建立詞彙表。
2. 用適配後的 SecureBERT 取得每個 token 的 base embedding。
3. 根據 template 內 token 的餘弦相似度反推權重。
4. 把每個 template 聚成一個向量，儲存成 `template_vectors.pkl`。

## 為什麼要先 fine-tune BERT

預訓練模型對系統呼叫樣板的語境不一定熟悉。`BertFinetuner` 的作用不是把模型訓練到很大，而是把語意空間往專案資料分佈稍微拉近，避免後面的 template 向量太漂移。

## 常見 fallback

- 若資料太少，直接保存原始 SecureBERT。
- 若遇到新 template，`TemplateEmbedder.get_vector()` 會即時計算並回填快取。
- 若 template 幾乎沒有可用 token，會回傳零向量作為保守 fallback。

## 相關

- [[TPLogAD-03-parsing-and-labels]]
- [[TPLogAD-05-para2vec-and-resource-encoding]]
- [[TPLogAD-09-failure-modes]]