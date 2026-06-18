# Model and Loss

這份文件整理模型與 loss 的契約。
它回答的不是「模型多深」，而是「模型要學什麼表示，以及 loss 如何把這些目標拉在一起」。

## 模型結構

`BiLSTMAttention` 的輸入是每個時間步的 `[template_vector, para_vector]` 拼接結果。它的主要路徑是：

1. `BiLSTM` 讀取序列上下文。
2. `AdditiveAttention` 把序列壓縮成 context vector。
3. `template_head` 預測下一個 template cluster。
4. `param_head` 重建當前時間步的參數向量。
5. `label_head`（若 `n_labels > 1`）進行監督式分類。

## 為什麼是雙任務

只有 template prediction 不一定能捕捉參數分佈，只有 reconstruction 也不一定能學到事件順序。雙任務的設計讓模型同時保留：

- 序列模式。
- 參數幾何結構。
- 可選的 label supervision。

## Loss 的組成

- `TemplateLoss`：Cross-Entropy，用來學下一個 template class。
- `ParamReconLoss`：MSE，用來重建 para2vec 向量。
- `LabelLoss`：Cross-Entropy，用來做 binary 或 technique 監督式分類。
- `CombinedLoss`：把上述 loss 加權合成。

### 權重直覺

- `alpha` 越大，模型越偏向 template classification。
- `alpha` 越小，模型越偏向參數重建。
- `label_weight` 越大，監督式分類對梯度影響越強。

## 推論時怎麼用

- `template_logits` 可以拿來算 template anomaly score。
- `param_pred` 可以和真實 para vector 算重建誤差。
- `attn_weights` 可以拿來做可視化與解釋。
- `label_logits` 若存在，則可做 supervised classification。

## 相關

- [[TPLogAD-01-offline-trainer-entrypoint]]
- [[TPLogAD-03-parsing-and-labels]]
- [[TPLogAD-08-cache-and-rerun]]