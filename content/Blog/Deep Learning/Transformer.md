---
type: Blog-post
tags:
  - Deep-Learning
  - note
created: 2026-05-27
updated: 2026-05-27
authors: WeiTa
status: active
---
---

# Transformer：從序列建模到通用表示框架

## 這篇文章想回答什麼問題？

這篇文章的起點是一個很基本但常被略過的問題：

> 序列模型為什麼一定要從左讀到右？

RNN 的邏輯是「先讀到 $t-1$ 步，才能算第 $t$ 步」，這很直覺，但也因此決定了它很難平行化。Transformer 在 2017 年的 *Attention Is All You Need* 裡給出了一個完全不同的答案：

**不需要**。

它把序列建模改寫成一個更泛用的問題：任意兩個 token 之間，根據內容動態決定彼此的關聯強度。這件事同時解決了 RNN 的速度限制，也讓 Transformer 幾乎成為現代深度學習的骨幹——從 BERT、GPT 到 ViT，再到現在這篇要延伸討論的 MaskGIT。

如果把這篇文章濃縮成三個問題：

1. Attention 是什麼，它和 RNN、CNN 的建模方式有什麼本質差異？
2. Transformer 的數學怎麼走，每一步的實作動機是什麼？
3. MaskGIT 怎麼把 Transformer 的 masked prediction 能力，移植到影像生成？

---

## 摘要
本文從問題切入——為什麼序列不必固守左到右——解釋 Attention 的直覺與數學，逐步引導讀者理解 Transformer 的關鍵設計（QKV、scaled dot-product、multi-head、位置編碼、殘差與層歸一化），並透過 ViT 與 MaskGIT 的實例，說明 Transformer 如何擴展到視覺任務與生成任務。


## 為什麼舊方法不夠好？

### RNN / LSTM：序列依賴是把雙刃劍

RNN 的狀態更新可以寫成：

$$
h_t = f(h_{t-1},\, x_t)
$$

這個設計很符合「記憶會隨時間演進」的直覺，但隱藏的代價是：每一步都綁死在前一步的輸出上。換句話說，不管 batch 多大、GPU 多快，RNN 的序列方向就是沒辦法平行。

長句子還有另一個問題。梯度要從最後一個時間步往回傳，經過幾十步之後很容易消失，導致模型根本學不到長距離依賴。LSTM 用 cell state 和各種 gate 緩解了一部分，但本質上還是在同一條路上繞。

### CNN：快，但看不遠

CNN 的平行化沒有問題，但它的感受野天生是局部的。若想讓第 1 個位置和第 100 個位置產生關聯，你要不堆很多層讓感受野逐漸擴大，要不換成 dilated convolution 手動拉大間距。

這兩種方式都不夠優雅，而且計算路徑很長，梯度同樣不好傳。

> **Insight：** 兩種模型各有硬傷。RNN 限速於時間依賴，CNN 限速於空間局部性。Transformer 要同時解決這兩件事。

---

## 核心想法：Attention 是動態查詢，不是記憶

要理解 Attention，先把它想像成一個資料庫查詢系統：

- **Query**：「我現在想找什麼」
- **Key**：「我身上有哪些可以被比對的特徵」
- **Value**：「如果你真的要看我，可以拿走哪些內容」

傳統 RNN 是把所有資訊壓進一個 hidden state，再靠 gate 決定要記住或忘記什麼。Attention 不一樣：它不預設哪個資訊重要，而是在每一次計算時，根據當前 query 和全部 key 的相似度，動態決定要把注意力放在誰身上。

這件事帶來兩個根本改變：

1. 任意兩個位置之間的距離，在 attention 的角度下都是一步。長距離依賴不再是問題。
2. Query 對 Key 的比對可以全部一起算，不需要等待前一步。計算可以平行化。

---

## 模型架構

Transformer 的標準流程可以拆成幾個清楚的階段：

```
token → embedding → + positional encoding
       ↓
  [Multi-Head Self-Attention]
       ↓
  [Add & LayerNorm]
       ↓
  [Feed-Forward Network]
       ↓
  [Add & LayerNorm]
       ↓
  repeat × N layers
       ↓
  Encoder output / Decoder output
```

每一層都在做同一件事：先讓每個 token 去看整個序列、更新自己的表示，再做一次非線性轉換整理特徵。

Encoder 把輸入壓成上下文表示，Decoder 則根據 Encoder 的輸出加上自回歸約束，逐步生成目標序列。

---

## 數學推導

### Step 1：把輸入向量化

文字進入模型之前，先透過 embedding 矩陣映射成向量。但 self-attention 本身對位置是盲目的——它只知道「有哪些 token」，不知道「它們排在什麼順序」。所以要在 embedding 上疊一個位置編碼：

$$
a_i = e_i + p_i
$$

其中 $e_i$ 是 token embedding，$p_i$ 是位置編碼。

### Step 2：Positional Encoding

原始論文使用正弦與餘弦函數構造固定位置編碼：

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

$pos$ 是位置索引，$i$ 是維度索引，$d$ 是 embedding 總維度。

這個設計可以理解成：每個位置被編成一組「多頻率波形指紋」，高維度對應低頻（全局位置），低維度對應高頻（局部細節）。不同位置之間的差可以被模型學會解讀，這就是它能表示相對位置關係的原因。

> **Note：** 近年的模型（如 RoPE、ALiBi）大多改用可學習或相對位置編碼，但原始正弦版本的直覺仍然值得理解。

### Step 3：Query / Key / Value 投影

對每一個輸入向量 $a_i$，用三組可訓練的線性矩陣做投影：

$$
q_i = W_q\, a_i, \qquad k_i = W_k\, a_i, \qquad v_i = W_v\, a_i
$$

三者都來自同一個輸入，這也是為什麼叫做 **self**-attention。

角色分工：$q_i$ 負責提問，$k_i$ 負責被比對，$v_i$ 負責提供實際內容。投影矩陣是可以訓練的，代表模型可以自己學到「什麼樣的問題配什麼樣的答案」。

### Step 4：Scaled Dot-Product Attention

計算第 $i$ 個 token 對其他所有 token 的關注分數：

$$
\alpha_{i,j} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

除以 $\sqrt{d_k}$ 是實作上的穩定技巧。當維度 $d_k$ 很大時，內積數值也會隨之放大，讓 softmax 的輸出極度集中在某幾個位置，梯度幾乎消失。縮放之後，分佈會更平滑，訓練也穩定得多。

接著把分數轉成機率分佈，再做加權和：

$$
\hat{\alpha}_{i,j} = \text{Softmax}\!\left(\alpha_{i,j}\right), \qquad
b_i = \sum_j \hat{\alpha}_{i,j}\, v_j
$$

$b_i$ 就是第 $i$ 個 token 更新後的表示——它不再只反映自己的原始 embedding，而是融合了整個序列根據語意關係傳遞過來的資訊。

### Step 5：矩陣化與平行運算

把整段序列堆成矩陣，三組投影可以一次完成：

$$
Q = AW_q, \qquad K = AW_k, \qquad V = AW_v
$$

完整的 attention 計算變成：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$QK^T \in \mathbb{R}^{N \times N}$ 這個矩陣每一格都是「第 $i$ 個 token 和第 $j$ 個 token 的關聯強度」。整段計算不依賴任何先後順序，GPU 可以一次全部算完。

這就是 Transformer 訓練快的核心原因。

---

## 多頭注意力（Multi-Head Attention）

單一 attention head 只能學到一種關係視角。但語言或影像裡的關係是多層次的：主詞動詞對應、指代詞回指、局部紋理依存……

Multi-head 的做法是讓模型同時跑 $h$ 個 head，每個 head 有自己獨立的投影矩陣：

$$
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

最後把所有 head 的輸出串接，再做一次線性整合：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O
$$

> **Insight：** 不同 head 可以分別捕捉句法關係、語義指代、局部共現等，類似讓多個專家同時看同一段輸入，再把意見整合起來。

---

## Tensor Shape / Data Flow

序列長度 $N$，embedding 維度 $d$：

| 階段 | Shape |
|------|-------|
| 輸入矩陣 $A$ | $\mathbb{R}^{N \times d}$ |
| $Q, K, V$ | $\mathbb{R}^{N \times d_k}$ |
| Attention score $QK^T$ | $\mathbb{R}^{N \times N}$ |
| Attention 輸出 | $\mathbb{R}^{N \times d}$ |

注意 Transformer 的輸出和輸入 shape 相同。每個位置都更新了自己的表示，但序列結構沒有被壓縮——這和 RNN 把整段歷史壓進 hidden state 有根本的不同。

---

## 殘差連接與層歸一化（Add & Norm）

Transformer 每個子層都包著 residual connection 和 layer norm：

$$
\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

**Residual connection** 讓梯度有一條更直接的通道往回傳，也讓深層的模型在訓練初期接近 identity mapping，比較容易收斂。

**Layer Normalization** 針對單一樣本在特徵維度上做標準化：

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

其中 $\mu, \sigma^2$ 是該 token 在特徵維度上的均值與變異數。它的效果是把不同層的數值分佈穩定在一個合理的範圍，讓損失地形更平坦，梯度下降不容易 zigzag。

---

## Decoder 與 Masked Self-Attention

生成任務有一個限制：預測第 $t$ 個詞時，只能用前面已生成的 $1 \sim t-1$ 個詞，不能偷看後面。

Transformer decoder 的做法是加上 causal mask：把 attention score 矩陣中「未來位置」的分數設成 $-\infty$，softmax 之後這些位置的權重就變成 0：

$$
\text{softmax}(-\infty) = 0
$$

視覺上就是一個下三角遮罩：每個位置只能看到自己和它左邊的 token。

此外，decoder 還有一個 **cross-attention**：Query 來自 decoder 自己，Key 和 Value 來自 encoder 的輸出。這讓 decoder 在生成時可以動態參考 encoder 處理好的原始輸入。

> 翻譯任務的簡化描述：Encoder 負責讀懂原文，Cross-Attention 讓 Decoder 知道要對應哪個部分，Masked Self-Attention 確保生成過程不作弊。

---

## BERT：Encoder-Only 的預訓練代表

BERT 把 Transformer 的 Encoder 拿出來，用大規模的無監督資料做預訓練，再 fine-tune 到各種下游任務上。

### Masked Language Model（MLM）

BERT 隨機遮蔽輸入中約 15% 的 token，讓模型預測被遮住的詞。遮蔽策略分三種：

- 80%：換成 `[MASK]`
- 10%：換成隨機 token
- 10%：保留原字不動

不全部換成 `[MASK]` 的原因，是要讓模型在 fine-tune 時不要對這個特殊符號過度依賴，學到更健壯的上下文表示。

### Fine-tuning

分類任務裡，BERT 取 `[CLS]` token 的最後一層 hidden state，接一個分類頭：

$$
P = \text{Softmax}(c\,W^T)
$$

其中 $c$ 是 `[CLS]` 的表示。BERT 的預訓練讓這個表示本身就富含語意，fine-tune 只需要少量標注資料。

---

## Vision Transformer（ViT）：把影像也變成序列

Transformer 的核心不依賴文字。只要把影像切成一塊一塊的 patch，每個 patch 就可以當成一個 token。

以 $224 \times 224 \times 3$ 的影像為例，切成 $16 \times 16$ 的 patch 後，會得到 $14 \times 14 = 196$ 個 patch。每個 patch 展平後做線性投影，加入一個可學習的 `[CLS]` token 和一維位置編碼，就能送進標準 Transformer Encoder。

ViT 說明了一件很重要的事：

> Transformer 不是 NLP 專屬，而是一種通用的關係建模框架。只要資料能被切成 token 序列，Attention 就能讓它們彼此溝通。

---

## MaskGIT：把 BERT 的 Masked Prediction 移植到影像生成

### 這個模型要解決什麼問題？

影像生成有兩條主流路線：

1. **自回歸生成（Autoregressive）**：像 GPT 一樣，一個 token 接一個 token 從左上到右下依序生成。品質不錯，但速度很慢——生成一張 $256 \times 256$ 的圖就要預測 $256$ 個 token，完全串行。
2. **擴散模型（Diffusion）**：品質高，但推論需要幾十步甚至幾百步的去噪。

MaskGIT（2022, Google Research）走的是第三條路：

> **用 BERT 的 masked prediction，搭配迭代解碼，在幾步內同時生成多個 token。**

### 架構全貌

MaskGIT 的架構可以分成兩個部分：一個用來把影像離散化的 VQGAN，以及一個用來做 masked prediction 的雙向 Transformer。

```
[Training]
  Image
    ↓ VQGAN Encoder (CNN)
  Discrete Token Map  (h × w 個 token，每個來自 codebook)
    ↓ Random Masking（masking ratio γ 隨機抽樣）
  Masked Token Sequence
    ↓ Bidirectional Transformer
  預測每個 [MASK] 位置對應的 codebook index
    ↓ Cross-Entropy Loss（只對 masked token 計算）

[Inference]
  全部 [MASK]
    ↓ Transformer 預測 → 取 confidence 最高的 k 個 token 解碼
  部分 [MASK]（剩下的繼續 mask）
    ↓ Transformer 預測 → 再解碼更多 token
  ...（重複 T 步）
  完整 Token Map
    ↓ VQGAN Decoder (CNN)
  Generated Image
```

### VQGAN：把影像壓成離散 Token

MaskGIT 不直接在像素空間操作，而是先用 VQGAN 把影像轉成一組離散的 token。

VQGAN 包含一個 CNN encoder $E$ 和一個 codebook $\mathcal{Z} = \{z_k\}_{k=1}^{K}$。輸入影像 $x$ 先被 encoder 壓成連續特徵圖，再對每個位置做 nearest-neighbor lookup，找出最近的 codebook vector：

$$
z_q = \arg\min_{z_k \in \mathcal{Z}} \|E(x)_{ij} - z_k\|_2
$$

這樣一張影像就被表示成一個離散的 index map，shape 為 $h \times w$，每個位置是一個 codebook index（整數）。把這個 index map 攤平，就得到長度 $N = h \times w$ 的 token 序列，作為 Transformer 的輸入。

> **Note：** 和 VQVAE 不同，VQGAN 加入了 perceptual loss 和 patch-based adversarial loss，讓重建品質更好，特別是高頻紋理細節。

### 雙向 Transformer：Bidirectional，而不是 Causal

這裡是 MaskGIT 和 GPT 最根本的架構差異。

GPT 用 causal mask，每個 token 只能看前面的位置。MaskGIT 的 Transformer 是雙向的，沒有任何 mask on attention——每個 `[MASK]` token 都可以看到整個序列所有未被遮蔽的 token。

這讓預測更準，因為影像的每個 patch 和周圍所有位置都有空間關聯，強行規定「從左到右」的生成順序對影像來說是沒有意義的。

### 訓練：Masked Visual Token Modeling（MVTM）

訓練流程直接對應 BERT 的 MLM，但用在離散的影像 token 上：

1. 對每個訓練樣本，從一個 masking ratio 分佈 $\gamma \sim p(\gamma)$ 中抽樣，決定要遮蔽幾個 token。
2. 被遮蔽的位置替換成特殊的 `[MASK]` embedding。
3. Transformer 對每個被遮蔽的位置，預測原本的 codebook index。
4. 只對被遮蔽的位置計算 cross-entropy loss。

論文裡的 masking ratio 分佈並非均勻，而是傾向讓比例在 $0$ 和 $1$ 之間都有出現，讓模型在各種遮蔽程度下都練習過。

### 推論：Iterative Parallel Decoding

這是 MaskGIT 最有趣的地方。生成不是一步到位，也不是 token-by-token，而是**迭代式並行解碼**。

定義一個 masking schedule $\gamma(t)$，控制在第 $t$ 步時還有多少比例的 token 要保持遮蔽。原始論文採用 cosine schedule：

$$
\gamma(t) = \cos\!\left(\frac{\pi\, t}{2T}\right), \quad t = 0, 1, \dots, T
$$

其中 $T$ 是總步數。當 $t=0$ 時 $\gamma=1$，全部都是 `[MASK]`；當 $t=T$ 時 $\gamma=0$，全部 token 都已解碼。

**每一步的流程：**

1. 對當前所有 `[MASK]` 位置，Transformer 同時預測每個位置的 token（以及對應的 confidence，即 softmax 機率的最大值）。
2. 根據 $\gamma(t+1)$ 計算下一步要保留多少 `[MASK]`，決定要解碼幾個 token。
3. 從 confidence 最高的位置開始解碼，其餘繼續保持 `[MASK]`。
4. 重複直到全部 token 都解碼完畢。

```
t=0:  [M][M][M][M][M][M][M][M]   ← 全部 mask
       ↓ Transformer 預測 + 取高 confidence
t=1:  [3][M][7][M][M][2][M][M]   ← 解碼部分 token
       ↓ Transformer 再預測（現在有更多上下文）
t=2:  [3][5][7][M][1][2][M][9]
       ↓
t=T:  [3][5][7][4][1][2][6][9]   ← 完整生成
```

> **Insight：** 先解碼 confidence 高的位置，讓後面的步驟有更好的上下文，再解碼比較難的位置。這和人類畫圖的直覺很像：先確定大結構，再填細節。

這個設計讓 MaskGIT 只需要幾步（論文裡預設 $T=8$）就能生成一張完整影像，比自回歸方法快了幾個數量級。

### MaskGIT 與 Transformer 的設計對應

| 概念 | Transformer（NLP） | MaskGIT（影像生成） |
|------|-------------------|-----------------|
| Token | 文字 token | VQGAN 離散 patch index |
| 遮蔽策略 | MLM，隨機遮蔽 | MVTM，隨機遮蔽 + schedule |
| Attention | Bidirectional（BERT）或 Causal（GPT） | 純 Bidirectional |
| 生成方式 | Autoregressive（GPT）或 one-shot（BERT fine-tune）| Iterative parallel decode |
| 位置編碼 | 1D 正弦或可學習 | 2D learnable positional embedding |

兩者最深層的共通點：都是在學「給定部分觀測，預測被遮蔽的 token」。差別只在 MaskGIT 把這個能力轉成了一個可以迭代的生成流程。

---

## 常見誤解

**Attention 不是「記住全部內容」**

Attention 是根據當前 query 動態分配權重，不是硬記所有資訊。它更像查詢系統，不是死背。

**Position encoding 不等於語意**

位置編碼只補順序，語意來自 embedding 和 attention 的交互學習。

**Multi-head 不是把模型切小**

多頭是讓不同 head 捕捉不同視角，增加表示能力，不是平分容量。

**MaskGIT 的 Transformer 不能換成 causal**

改成 causal mask 之後就回退成自回歸模型，失去了雙向上下文的優勢，生成速度也會變慢。

---

## 與其他模型的比較

| | RNN | CNN | Transformer | MaskGIT |
|--|-----|-----|-------------|---------|
| 序列建模 | 串行 | 局部 | 全域 Attention | 雙向全域 |
| 長距離依賴 | 難 | 需堆層 | 容易 | 容易 |
| 訓練平行化 | 困難 | 容易 | 容易 | 容易 |
| 生成速度 | — | — | Token-by-token（GPT）| 幾步並行 |
| 影像生成 | 不適合 | 需特殊設計 | ViT（理解任務）| 原生生成任務 |

---

## 總結

1. **Transformer 解決的是建模方式的問題，不只是速度問題。** Attention 讓任意兩個位置可以直接通訊，打破了 RNN 的串行限制和 CNN 的局部限制。
2. **每個設計都有實作理由。** Scaled dot-product 防梯度消失，Multi-head 增加視角多樣性，Residual + LayerNorm 讓深層訓練穩定。
3. **MaskGIT 是 BERT 邏輯在影像生成的延伸。** 用 VQGAN 離散化影像、用雙向 Transformer 做 masked prediction、用 iterative decoding 讓生成可以在幾步內完成。它的代價是需要 VQGAN 的重建品質，以及 codebook 的設計。
4. **Transformer 的本質是通用框架，不只是 NLP 工具。** 只要能把資料切成 token，attention 就能學關係。ViT 和 MaskGIT 都是這個想法的實踐。

---

## Takeaways（關鍵帶走）

- **Attention 作為通用關係建模工具**：不再依賴序列順序，任意兩個 token 可以直接互動，這是 Transformer 的核心勝算。
- **工程設計的互補性**：位置編碼、scaled dot-product、multi-head、殘差與 LayerNorm 等設計各司其職，共同維持訓練穩定與表現。
- **從文字到影像的橋樑**：ViT 與 MaskGIT 展示了把資料切成 token 的普適性，但在影像任務上還需處理重建品質（VQGAN/codebook）與 2D positional inductive bias。
- **實務建議**：學習 Transformer 時，先理解 Scaled Dot-Product Attention 的矩陣化形式與數值穩定技巧，再從小模型實作（單 head、單層）逐步加複雜度。
