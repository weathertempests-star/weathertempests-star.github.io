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

# RNN：讓神經網路學會「記憶」

## 這篇文章想回答什麼問題？

這篇文章的起點是一個很基本的問題：

> 如果 MLP 已經能逼近任意函數，為什麼它沒辦法好好讀一個句子？

這個問題牽出了四條主線：

1. 序列資料和一般向量資料有什麼本質差異，MLP 為什麼天生不適合？
2. RNN 怎麼用「共享參數 + 遞迴狀態」解決這個問題？
3. 反向傳播一旦穿越時間軸，梯度會發生什麼事？梯度消失和梯度爆炸從哪裡來？
4. LSTM 用什麼機制修補 vanilla RNN 的記憶瓶頸？

如果把整篇濃縮成一句話：

> RNN 把序列建模改寫成「用固定參數反覆更新記憶」的問題；
> 但這個設計讓梯度在時間軸上連乘，導致長距離依賴幾乎學不動；
> LSTM 透過 gate 機制讓模型自己決定該記什麼、忘什麼，才真正讓記憶變得可控。

---

## 為什麼 MLP 不適合序列資料？

MLP 的輸入是一個固定長度的向量，所有位置都被當成**平等、無順序**的特徵。但序列資料根本不長這樣。

### 序列資料的三個本質特性

**第一，順序本身就是資訊。** 「狗咬人」和「人咬狗」用的是同樣的詞，但意思截然不同。如果把 token 展平成向量再丟進 MLP，順序就消失了。

**第二，輸入長度是變動的。** MLP 的輸入維度是固定的，但句子、音訊、時間序列，長度在訓練和測試時都可能不一樣。

**第三，當前位置的意義依賴前文。** 閱讀一個詞時，你不只看這個詞本身，還要記住前面讀過什麼。MLP 沒有跨位置的「記憶」機制。

這三個問題都指向同一件事：MLP 把每筆輸入看成**獨立、無結構的點**，但序列資料需要的是**有記憶的、帶時間感的閱讀方式**。

> **Insight：** MLP 的弱點不是表達能力不夠，而是歸納偏差（inductive bias）選錯了。影像的問題是空間局部性，序列的問題是時間依賴性。CNN 解決前者，RNN 解決後者。

---

## 核心想法：把歷史壓進一個向量

RNN 的基本想法很直觀：**不要一次把整段序列丟進去，而是一個時間步一個時間步地讀，同時維護一個會跟著更新的「記憶」。**

這個記憶，就是 **hidden state** $h^{(t)}$。

形式上，RNN 是一個動態系統：

$$
h^{(t)} = f_\theta\!\bigl(h^{(t-1)},\, x^{(t)}\bigr)
$$

每個時間步的 hidden state，由**上一步的記憶**和**當前輸入**共同決定，函數 $f_\theta$ 在每個時間步完全共享同一組參數 $\theta$。

把這件事攤開成 unfolded computational graph，就能清楚看到 RNN 在做什麼：

```
x(1)    x(2)    x(3)    ...    x(T)
  |       |       |               |
h(0) → h(1) → h(2) → h(3) → ... → h(T)
         |       |       |               |
       o(1)   o(2)   o(3)         o(T)
```

三件事值得注意：

1. **同一組參數在每個時間步都重複使用**，讓模型對變長序列有泛化能力。
2. **$h^{(t)}$ 承載了從 $x^{(1)}$ 到 $x^{(t)}$ 的整段歷史摘要**，但摘要的品質取決於 $h$ 的維度和模型能學到多少。
3. **能處理任意長度**，不等於**能記住任意長的上下文**。這是 RNN 最關鍵的結構限制。

> **Warning：** 「RNN 可以處理任意長度的序列」是一個常被誤解的說法。它的確可以逐步讀入任意長的輸入，但 hidden state 的維度是固定的，所以長序列仍然會有信息丟失。能處理和能記住，是兩件不同的事。

---

## 模型架構：Vanilla RNN 的標準形式

### 前向傳播

標準 RNN 的計算可以拆成三步：

**Step 1 — 更新 hidden state：**

$$
a^{(t)} = b + W h^{(t-1)} + U x^{(t)}
$$

$$
h^{(t)} = \tanh\!\bigl(a^{(t)}\bigr)
$$

**Step 2 — 產生輸出：**

$$
o^{(t)} = c + V h^{(t)}
$$

**Step 3 — 轉成機率（如果是分類任務）：**

$$
\hat{y}^{(t)} = \text{softmax}\!\bigl(o^{(t)}\bigr)
$$

三組可訓練參數的角色：

| 參數 | Shape | 語意 |
|------|-------|------|
| $U \in \mathbb{R}^{d_h \times d_x}$ | input-to-hidden | 當前輸入怎麼影響記憶 |
| $W \in \mathbb{R}^{d_h \times d_h}$ | hidden-to-hidden | 舊記憶怎麼傳遞到新記憶 |
| $V \in \mathbb{R}^{d_y \times d_h}$ | hidden-to-output | 記憶怎麼轉成輸出 |
| $b \in \mathbb{R}^{d_h}$、$c \in \mathbb{R}^{d_y}$ | bias | — |

### Tensor Shape / Data Flow

以 batch 維度追蹤一次完整的前向傳播（batch size $B$，序列長度 $T$，輸入維 $d_x$，隱藏維 $d_h$，輸出維 $d_y$）：

```
x^(t):      [B, d_x]
h^(t-1):    [B, d_h]

a^(t) = b + W @ h^(t-1)       +   U @ x^(t)
              ↑                        ↑
   [d_h,d_h]@[B,d_h]=[B,d_h]   [d_h,d_x]@[B,d_x]=[B,d_h]

h^(t) = tanh(a^(t)):           [B, d_h]
o^(t) = c + V @ h^(t):         [B, d_y]
y_hat^(t) = softmax(o^(t)):    [B, d_y]
```

整段序列的 hidden state 堆起來就是 $[T, B, d_h]$，輸出就是 $[T, B, d_y]$。

---

## 數學推導

### 損失函數：把序列拆成每步貢獻的加總

如果每個時間步都有對應的標籤 $y^{(t)}$（例如語言模型預測下一個詞），整段序列的損失就是各步損失的總和：

$$
L\bigl(\{x^{(1)},\dots,x^{(T)}\},\, \{y^{(1)},\dots,y^{(T)}\}\bigr) = \sum_{t=1}^T L^{(t)}
$$

單步損失是負對數概似：

$$
L^{(t)} = -\log p_{\text{model}}\!\bigl(y^{(t)} \mid x^{(1)}, \dots, x^{(t)}\bigr)
$$

> **Note：** 這裡有一個隱藏的因果性限制：$t$ 步的預測只能用到 $t$ 步以前的資訊，不能偷看未來。在 unfolded graph 裡，這件事自然成立，因為 $h^{(t)}$ 只依賴 $h^{(t-1)}$ 與 $x^{(t)}$。

### BPTT：把反向傳播攤開到時間軸上

**Backpropagation Through Time（BPTT）** 就是把 RNN 的 unfolded graph 當成一個普通的計算圖，然後從後往前一步步地用 chain rule 算梯度。

#### Step 1：輸出層的梯度

對 softmax + cross-entropy 而言，單步損失對輸出的梯度有漂亮的形式：

$$
\nabla_{o^{(t)}} L = \hat{y}^{(t)} - \mathbf{y}^{(t)}
$$

其中 $\mathbf{y}^{(t)}$ 是 one-hot 標籤向量。

#### Step 2：hidden state 的梯度遞迴

每個 $h^{(t)}$ 的梯度來自兩條路：

- 往右：對當前時間步的輸出 $o^{(t)}$ 的貢獻
- 往後：對下一個時間步 $h^{(t+1)}$ 的貢獻

寫成遞迴式：

$$
\nabla_{h^{(t)}} L
=
\underbrace{V^T \nabla_{o^{(t)}} L}_{\text{來自輸出}}
+
\underbrace{W^T H^{(t+1)} \nabla_{h^{(t+1)}} L}_{\text{來自下一步}}
$$

其中 $H^{(t+1)}$ 是對角矩陣，對角元素是 $\tanh$ 的導數：

$$
H^{(t+1)}_{ii} = 1 - \bigl(h_i^{(t+1)}\bigr)^2
$$

邊界條件：在最後一個時間步 $T$，只有來自輸出的梯度，沒有來自下一步的部分。

#### Step 3：參數梯度累加

參數在每個時間步都被重複使用，所以它的梯度要把所有時間步的貢獻加起來：

$$
\nabla_W L = \sum_{t=1}^T H^{(t)} \bigl(\nabla_{h^{(t)}} L\bigr) \bigl(h^{(t-1)}\bigr)^T
$$

$$
\nabla_U L = \sum_{t=1}^T H^{(t)} \bigl(\nabla_{h^{(t)}} L\bigr) \bigl(x^{(t)}\bigr)^T
$$

$$
\nabla_V L = \sum_{t=1}^T \bigl(\nabla_{o^{(t)}} L\bigr) \bigl(h^{(t)}\bigr)^T
$$

> **Insight：** 這裡可以看到 RNN 和 MLP 最根本的訓練差異：MLP 的參數只在一個位置出現，而 RNN 的 $W$、$U$ 在 $T$ 個位置全部出現，所以梯度要加總。每個 epoch 做的是一次大規模的時間展開圖上的 backpropagation。

---

## 梯度消失與梯度爆炸

BPTT 暴露了 RNN 最核心的訓練困境。

把梯度從時間步 $t$ 往回傳到時間步 $k$（$k < t$），中間要乘上 $t-k$ 個轉移矩陣和 $\tanh$ 導數：

$$
\frac{\partial h^{(t)}}{\partial h^{(k)}} = \prod_{i=k+1}^{t} \frac{\partial h^{(i)}}{\partial h^{(i-1)}} = \prod_{i=k+1}^{t} W^T H^{(i)}
$$

這是一個**矩陣的連乘積**。連乘積的數值行為完全由矩陣的特徵值決定：

$$
\left\|\frac{\partial h^{(t)}}{\partial h^{(k)}}\right\| \approx |\lambda|^{t-k}
$$

- 若最大特徵值 $|\lambda| < 1$：隨著距離 $t-k$ 增大，梯度指數衰減 → **梯度消失（vanishing gradient）**
- 若最大特徵值 $|\lambda| > 1$：隨著距離增大，梯度指數爆炸 → **梯度爆炸（exploding gradient）**

### 兩種問題的對症下藥

| 問題 | 現象 | 解法 |
|------|------|------|
| 梯度消失 | 早期時間步的參數學不動，長距離依賴無法學習 | LSTM/GRU 的 gate 機制、殘差連接 |
| 梯度爆炸 | loss 劇烈震盪，參數爆炸，甚至出現 NaN | Gradient Clipping |

### Gradient Clipping 的實作

梯度爆炸的解法很直接：當梯度的 norm 超過閾值 $v$ 時，把它縮放回來：

$$
g \leftarrow \begin{cases}
g & \|g\| \le v \\[4pt]
\dfrac{g \cdot v}{\|g\|} & \|g\| > v
\end{cases}
$$

這個操作只縮放梯度的大小，不改變方向，是實作上幾乎必用的防護措施。

> **Note：** 梯度爆炸很容易診斷（loss 直接發散或出現 NaN），但梯度消失很難察覺——模型只是「沒有學到長距離依賴」，但損失看起來可能還在慢慢下降。這是訓練 RNN 最棘手的地方。

---

## 為什麼常用 tanh，而不是 sigmoid？

激勵函數的選擇直接影響梯度保留程度。比較兩者的導數範圍：

$$
\sigma'(x) = \sigma(x)(1-\sigma(x)) \in (0,\; 0.25]
$$

$$
\tanh'(x) = 1 - \tanh^2(x) \in (0,\; 1]
$$

`tanh` 的最大導數是 1，`sigmoid` 的最大導數只有 0.25。每過一個時間步，sigmoid 的梯度最多被乘以 0.25，過幾步就會消失得差不多。`tanh` 在 $x=0$ 附近的導數接近 1，梯度保留程度好很多。

另一個好處是 `tanh` 的輸出**中心化在 0**，不像 sigmoid 輸出在 $[0,1]$，因此 hidden state 的數值分佈比較穩定，有助於後續的矩陣運算。

---

## LSTM：用 Gate 讓記憶可控

Vanilla RNN 的問題不是「無法記憶」，而是「無法主動控制記憶」——它只能靠 $W$ 矩陣被動地傳遞資訊，卻不能有選擇地保留或清除。

LSTM（Long Short-Term Memory）的核心想法是：**讓模型自己學會該記什麼、忘什麼、輸出什麼。**

這靠三組 **gate** 和一條額外的 **cell state** $c^{(t)}$ 實現。Gate 的輸出都是透過 sigmoid 壓進 $[0,1]$ 的向量，語意上就是「保留多少比例」。

### 三個 Gate + Cell State

**Forget Gate**：決定從上一步的 cell state 中丟掉哪些資訊：

$$
f^{(t)} = \sigma\!\bigl(W_f h^{(t-1)} + U_f x^{(t)} + b_f\bigr)
$$

**Input Gate + 候選記憶**：決定要把哪些新資訊寫進 cell state：

$$
i^{(t)} = \sigma\!\bigl(W_i h^{(t-1)} + U_i x^{(t)} + b_i\bigr)
$$

$$
\tilde{c}^{(t)} = \tanh\!\bigl(W_c h^{(t-1)} + U_c x^{(t)} + b_c\bigr)
$$

**Cell State 更新**：結合遺忘和寫入：

$$
c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}
$$

**Output Gate**：決定從 cell state 讀出哪些資訊當作 hidden state：

$$
o^{(t)} = \sigma\!\bigl(W_o h^{(t-1)} + U_o x^{(t)} + b_o\bigr)
$$

$$
h^{(t)} = o^{(t)} \odot \tanh\!\bigl(c^{(t)}\bigr)
$$

### 為什麼 LSTM 緩解了梯度消失？

Cell state $c^{(t)}$ 的更新式：

$$
c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}
$$

是**加法**，而不是 vanilla RNN 的矩陣乘法。加法讓梯度可以直接「穿過」時間步，不會因為矩陣特徵值過小而衰減：

$$
\frac{\partial c^{(t)}}{\partial c^{(t-1)}} = f^{(t)}
$$

只要 forget gate $f^{(t)}$ 接近 1，梯度就能幾乎無損地往回傳。

> **Insight：** ResNet 的 skip connection 和 LSTM 的 cell state，都是「加法而非乘法」的梯度高速公路。設計思路不謀而合：讓梯度有一條可以直接通過的捷徑，而不是被迫穿越每一層的 Jacobian 矩陣。

---

## 訓練時實際發生什麼？

### Teacher Forcing

訓練語言模型時，有一個非常常見但容易引發問題的技巧叫 **Teacher Forcing**：在訓練時，每個時間步的輸入 $x^{(t+1)}$ 直接用**真實標籤** $y^{(t)}$，而不是模型自己預測出來的 $\hat{y}^{(t)}$。

好處是：梯度穩定、訓練快。

壞處是：測試時沒有真實標籤可以用，模型只能吃自己的預測，這造成了**訓練和推理之間的分佈差距（exposure bias）**。一旦某一步預測錯了，後面的步驟都要基於錯誤的輸入，錯誤會越滾越大。

### Truncated BPTT

真實序列可能非常長（例如整篇文章），把梯度一路傳回去計算成本極高。**Truncated BPTT** 的做法是把序列切成若干段，每段只做有限步的反向傳播，但 hidden state 的值在段與段之間保留下來繼續傳遞。

這是計算效率和梯度正確性之間的妥協。

### Sequence-to-Sequence：輸入輸出長度不同怎麼辦？

標準 RNN 假設每個時間步都有輸出，但很多任務是輸入和輸出長度不同：機器翻譯、文字摘要、語音辨識。

**Encoder-Decoder（Seq2Seq）** 架構的做法是：

- **Encoder**：讀入整段輸入序列，把所有資訊壓進最後一個 hidden state $h^{(T)}$（context vector）。
- **Decoder**：從 context vector 出發，逐步自回歸地生成輸出序列。

```
輸入：x(1) → x(2) → ... → x(T)
                             ↓
                    h(T)  [context vector]
                             ↓
輸出：y(1) → y(2) → ... → y(T')   ← Decoder 逐步生成
```

這個設計的根本缺陷：**無論輸入多長，所有資訊都要壓進固定大小的 context vector**。這正是後來 Attention 機制要解決的瓶頸。

---

## 常見誤解

**誤解 1：參數共享讓 RNN 不夠靈活。**
錯。共享參數讓同一套規則可以在不同時間步重複使用，反而讓模型對變長序列有泛化能力。這是 RNN 相比 MLP 的核心優勢之一。

**誤解 2：RNN 能處理任意長度，所以它能記住任意長的上下文。**
不是。Hidden state 是固定維度的向量，不管序列多長，資訊都要壓進去。長序列的早期資訊往往被後來的資訊覆蓋，無法保留。

**誤解 3：LSTM 完全解決了梯度消失。**
「緩解」是準確的說法，「解決」太強了。LSTM 透過 cell state 的加法結構讓梯度更容易傳遞，但當序列非常長，或 forget gate 持續接近 0 時，梯度仍然可能消失。

**誤解 4：時間步的索引本身對 RNN 有意義。**
標準 RNN 對 absolute time step 沒有直接依賴。它只看 $h^{(t-1)}$ 和 $x^{(t)}$，所以如果狀態一樣、輸入一樣，輸出就一樣，不管這是第幾步。

---

## 與其他模型的比較

| | MLP | CNN | RNN / LSTM | Transformer |
|--|-----|-----|------------|-------------|
| 序列建模 | 不適合 | 局部窗口 | 天生適合，逐步讀 | 全域 Attention |
| 長距離依賴 | 需手工特徵 | 需堆很多層 | 難（vanilla），LSTM 緩解 | 容易 |
| 訓練平行化 | 容易 | 容易 | **困難**（時間依賴） | 容易 |
| 記憶機制 | 無 | 無 | Hidden state（固定維度） | 動態 Attention，無固定記憶 |
| 變長輸入 | 不支援 | 需特殊設計 | 天然支援 | 天然支援 |

RNN 在平行化上的限制是它最大的致命傷。因為每個時間步的 $h^{(t)}$ 必須等 $h^{(t-1)}$ 算完才能算，整個序列方向就是無法並行。這也是 Transformer 為什麼能取代 RNN 的核心原因之一。

---

## RNN 到 Transformer 的演化脈絡

回頭看整個序列建模的發展，其實是一連串對瓶頸的修補：

```
Vanilla RNN
  → 問題：梯度消失、長距離依賴幾乎學不到
  → 修補：LSTM / GRU（gate 控制記憶的存取）

LSTM
  → 問題：context vector 壓縮瓶頸、序列無法並行
  → 修補：Seq2Seq + Attention（動態查詢 encoder 的各步輸出）

Seq2Seq + Attention
  → 問題：仍然是序列結構，訓練慢，難擴展
  → 修補：Transformer（把 Attention 做到底，完全不用遞迴）
```

每一代模型都在問：「上一個設計的瓶頸在哪裡，能不能不做這件事？」

Transformer 的答案是：**遞迴這件事本身不是必要的。** 只要讓每個 token 都能直接和所有其他 token 互動，就不需要靠 hidden state 一步一步傳資訊了。

---

## 總結

1. **RNN 解決的是 MLP 無法建模序列資料的問題**：它用共享參數 + hidden state 遞迴，讓同一套邏輯可以重複作用在任意長的序列上。

2. **Vanilla RNN 的數學由三個矩陣 $U, W, V$ 描述**，前向傳播是反覆更新 $h^{(t)} = \tanh(Wh^{(t-1)} + Ux^{(t)} + b)$，訓練靠 BPTT 把梯度沿時間軸反傳，參數梯度是所有時間步貢獻的總和。

3. **BPTT 的梯度是矩陣連乘積**，特徵值決定梯度的命運：小於 1 就消失，大於 1 就爆炸。Gradient Clipping 對付爆炸，LSTM 對付消失。

4. **LSTM 的核心創新是 cell state + gate 機制**：用 forget gate 控制遺忘、input gate 控制寫入、output gate 控制讀出。Cell state 的加法更新讓梯度有高速公路，這和 ResNet skip connection 的設計思路本質上一樣。

5. **RNN 最大的結構性限制是無法平行化**，以及 context vector 的固定維度壓縮瓶頸。前者讓 Transformer 取代了 RNN，後者讓 Attention 機制取代了固定 context vector。
