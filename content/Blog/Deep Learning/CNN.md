---
type: Blog-post
tags:
  - Deep-Learning
  - note
created: 2026-05-26
updated: 2026-05-27
authors: WeiTa
status: active
---

---


## 這篇文章想回答什麼問題？

MLP 理論上可以逼近任意函數，那為什麼放到影像任務上就顯得力不從心？

這個問題其實牽出了四條主線：

1. 影像資料的結構和一般向量資料有什麼本質差異？
2. CNN 的三個設計原則怎麼剛好對上這些差異？
3. 網路越深越好，為什麼深到一定程度反而會退化？ResNet 如何修補這件事？
4. 分類只需要一個標籤，分割需要每個像素都有答案——UNet 的 encoder-decoder 如何做到？

如果把整篇濃縮成一句話：

> CNN 是把「影像的空間結構假設」直接燒進架構的設計；
> ResNet 修掉了深度帶來的訓練問題；
> UNet 則把 CNN 抽出的特徵用 skip connection 接回來，讓空間精度不在下採樣的過程中被丟光。

---

**摘要（精華）**：本篇梳理 CNN 的直覺與工程設計，說明為何局部連接、權重共享與層級表示是處理影像的合理選擇，並說明 ResNet 與 UNet 在深層訓練與像素級預測上的關鍵改進與實作要點。

***

## 為什麼 MLP 不適合影像？

在深度學習出現之前，影像分類是「手工特徵 + 傳統分類器」的流程：先跑 SIFT、HOG、顏色直方圖把影像壓成向量，再丟給 SVM 或 Logistic Regression 做分類。這個流程的瓶頸不在分類器，而在特徵——不同任務要重新設計特徵，分類效果完全取決於設計者的直覺。

那為什麼不直接用 MLP？把像素展平成向量，不就解決了？

問題有三個，而且都很致命。

**第一，參數量爆炸。** 一張 224×224 的 RGB 影像展平後有 $224 \times 224 \times 3 = 150{,}528$ 個輸入。第一層若有 1024 個神經元，光這一層就要 $1.5 \times 10^8$ 個參數。這還只是第一層。

**第二，空間結構被丟掉了。** 展平之後，像素和像素之間的位置關係就消失了。「左邊的像素緊鄰右邊的像素」這件事，MLP 完全看不到。

**第三，沒有平移不變性。** 同一隻貓出現在圖片左上角或右下角，語意完全一樣，但對 MLP 來說它會學成兩組不同的參數。這既浪費，也不對。

這三個問題指向同一個結論：影像的問題不是「非線性不夠」，而是「歸納偏差 (inductive bias) 選錯了」。MLP 把所有輸入位置都當成平等的、無結構的，但影像本來就有局部相關性、模式重複性，以及空間層次結構。

---

## CNN 的三個核心設計原則

### 1. 局部連接 (Local Connectivity)

卷積核每次只看輸入的一小塊，而不是整張圖。這背後的假設是：

> 影像中的資訊是局部組織的。邊緣、角點、紋理，都是鄰近像素之間的關係，而不是遠端像素的組合。

一個 $3 \times 3$ 的 kernel 只有 9 個參數，但可以在整張圖上滑動，在每個位置做同樣的局部模式偵測。相比之下，MLP 需要為每個輸入位置都學一套獨立的參數。

### 2. 權重共享 (Weight Sharing)

同一組 kernel 在不同位置重複使用。這讓模型不需要為「左上角的水平邊緣」和「右下角的水平邊緣」各自學一套參數——邊緣就是邊緣，不管它在哪裡。

這個設計帶來的副作用是*平移等變性 (translation equivariance)*：輸入平移多少，feature map 也會對應平移多少，而不是輸出完全不同的東西。池化再把等變性進一步轉成近似的*不變性 (invariance)*。

### 3. 層級特徵組合 (Hierarchical Feature Composition)

單層卷積只能看到很小範圍的模式。把多層卷積堆疊起來，低層看到邊緣，中層把邊緣組合成形狀，高層再把形狀組合成物體。這個「從簡單到抽象」的過程，正好對應了人類視覺系統 V1 → V2 → V4 → IT 的層級結構。

Hubel 與 Wiesel 在 1960 年代的研究就已經發現，視覺皮層裡的神經元只對局部感受野（receptive field）內的刺激有反應，而且不同神經元對不同方向的邊緣敏感。CNN 幾乎是把這個發現直接工程化了。

---

## 數學：卷積到底在算什麼？

### Cross-correlation vs. Convolution

先釐清一個常見誤解：深度學習框架裡的「卷積」，其實是 cross-correlation，不是數學上嚴格的卷積。差別在於 kernel 有沒有先翻轉。

數學上的 2D 卷積（kernel 翻轉）：

$$
Y(i, j) = \sum_{m}\sum_{n} X(i - m,\; j - n)\, K(m, n)
$$

深度學習框架實作的 cross-correlation（kernel 不翻轉）：

$$
Y(i, j) = \sum_{m}\sum_{n} X(i + m,\; j + n)\, K(m, n)
$$

因為 kernel 是訓練出來的，翻不翻轉只是學到的 kernel 長得不一樣，對模型能力沒有影響。所以兩者混用在實務上問題不大，但理解上要清楚。

### 多通道卷積

RGB 影像不是二維，而是三維張量。一個輸出通道對應一組 kernel，這組 kernel 會跨過所有輸入通道做加總：

$$
Y(i, j, c_{out}) = \sum_{c_{in}} \sum_{m} \sum_{n} X(i+m,\; j+n,\; c_{in})\, K(m, n,\; c_{in},\; c_{out})
$$

若輸入是 $X \in \mathbb{R}^{H \times W \times C_{in}}$，kernel 是 $K \in \mathbb{R}^{k_H \times k_W \times C_{in} \times C_{out}}$，輸出就是 $Y \in \mathbb{R}^{H' \times W' \times C_{out}}$。

這裡最容易搞混的是：一個輸出通道不只看一個輸入通道，它會同時整合所有輸入通道的訊號，再加總成一個 feature map。

### Tensor Shape 的完整流動

以一個典型的分類 CNN 為例，tensor 長這樣流動：

```
輸入         [B, 224, 224, 3]
Conv(64)     [B, 224, 224, 64]    <- same padding
MaxPool      [B, 112, 112, 64]    <- 2x2, stride 2
Conv(128)    [B, 112, 112, 128]
MaxPool      [B, 56, 56, 128]
Conv(256)    [B, 56, 56, 256]
MaxPool      [B, 28, 28, 256]
GAP          [B, 256]             <- Global Average Pooling
FC + Softmax [B, num_classes]
```

其中 $B$ 是 batch size，每一次 MaxPool 把空間尺寸減半，通道數則隨層增加。

---

## 卷積的各種變體

### Stride

Stride 控制 kernel 每次移動多少格。Stride = 2 時，feature map 的空間尺寸直接減半，效果類似 pooling，但可以同時做特徵提取和下採樣，省掉一個步驟。現代架構（如 ResNet）傾向用 stride convolution 取代 pooling。

### Padding

- **Valid convolution**：不補零，輸出尺寸縮小。$H_{out} = H - k + 1$。
- **Same convolution**：補零使輸出尺寸不變。$H_{out} = H$。
- **Full convolution**：補更多零，輸出尺寸反而變大，轉置卷積（Transposed Convolution）用的就是這個邏輯。

### 感受野 (Receptive Field)

感受野是指 feature map 上某個位置能「看到」的輸入範圍。每疊一層 $3 \times 3$ 卷積，感受野增加 2：

| 層數 | 感受野大小 |
|------|-----------|
| 1    | 3×3       |
| 2    | 5×5       |
| 3    | 7×7       |
| $L$  | $(2L+1) \times (2L+1)$ |

所以三層 $3 \times 3$ 等效於一層 $7 \times 7$ 的感受野，但參數量是 $3 \times (3^2) = 27$ vs. $7^2 = 49$，前者少了快一半，還多了兩次非線性。這就是 VGG 的核心設計思路。

### Dilation（擴張卷積）

Dilation 在 kernel 內部插入間隔，讓同一個小 kernel 看見更大的範圍，但不增加參數數量。Dilation rate = $d$ 時，感受野從 $k$ 擴大到 $k + (k-1)(d-1)$。

$$
Y(i, j) = \sum_{m}\sum_{n} X(i + d \cdot m,\; j + d \cdot n)\, K(m, n)
$$

DeepLab 系列大量使用 dilated convolution，讓模型在不做 pooling（避免解析度損失）的情況下擴大感受野。

---

## 池化與非線性

### Pooling 的真正用途

Pooling 不是為了讓模型「看起來更複雜」，而是在做**局部統計摘要**，讓模型對微小的位置偏移、旋轉、形變有更高容忍度。

Max pooling 取局部最大值，語意是「這個區域裡有沒有某種特徵存在」；Average pooling 取平均，語意是「這個區域的特徵有多強」。前者在辨識任務裡通常更有效，因為物體存在與否比平均強度更重要。

### Conv → BN → ReLU

現代 CNN 的最小單元幾乎固定是這個組合：

1. **Conv**：做線性的局部特徵抽取。
2. **Batch Normalization**：把輸出正規化到均值 0、方差 1，穩定訓練過程，讓梯度不容易消失或爆炸。
3. **ReLU**：引入非線性，讓多層堆疊不退化成單一線性變換。

BN 的作用可以直覺理解成：把損失地形修平一點，讓梯度下降不容易走彎路。

---

## 架構演化：從 LeNet 到 VGG

### LeNet-5（1998）

LeNet-5 是第一個有實際意義的 CNN，用在手寫數字辨識。它的結構就是 Conv → Subsampling → Conv → Subsampling → FC，規模很小，但確立了「特徵抽取 + 分類」的分工方式。

### AlexNet（2012）

AlexNet 在 ImageNet 上把 top-5 錯誤率從 26% 降到 15%，直接引爆深度學習熱潮。它的關鍵貢獻是把 ReLU、Dropout、資料擴增、多 GPU 訓練這些實作技巧全部組合在一起，證明深層 CNN 在大規模資料上的有效性。

### VGGNet（2014）

VGG 的策略極其簡單：只用 $3 \times 3$ 卷積，然後不斷堆疊。VGG-16 有 16 層，VGG-19 有 19 層。它的貢獻是確立了「深度很重要，kernel 要小」的設計原則，但參數量仍然巨大（138M），推理速度也慢。

更重要的是，VGG 之後有人問了一個問題：

> 既然深度是關鍵，那為什麼超過一定深度，模型在訓練集上的表現反而開始下降？

---

## ResNet34：為什麼深度網路會退化，以及如何修補

### 問題：深度退化（Degradation Problem）

直覺上，網路越深，表達能力越強，至少理論上不應該比淺層的版本更差。但實驗結果告訴我們，56 層的普通 CNN 在訓練集上的錯誤率比 20 層更高。這不是 overfitting（測試集也更差），而是優化問題——深層網路在訓練時根本學不到有效的東西。

原因出在梯度。網路太深時，梯度從 loss 往回傳，每經過一層都要乘上一個數，如果這個數小於 1，乘到最後梯度就接近 0，底層的參數完全學不動，這就是**梯度消失（Vanishing Gradient）**。

### 解法：Residual Learning

He et al.（2015）提出的想法很直覺：

> 與其讓一組層學 $H(x)$，不如讓它學 $F(x) = H(x) - x$，也就是殘差。

這樣輸出變成：

$$
H(x) = F(x) + x
$$

其中 $x$ 是 skip connection 直接繞過這組層的輸入。

為什麼這樣有用？因為如果這組層什麼都不學（$F(x) = 0$），輸出直接就是恆等映射 $H(x) = x$。這個「什麼都不做」的極端情況對一般 CNN 來說很難學到，但對 ResNet 來說只需要把 $F(x)$ 學成 0 就好了。換句話說，**學殘差比學完整映射更容易初始化到一個合理的起點**。

更關鍵的是梯度流動。Skip connection 提供了一條梯度的高速公路，讓梯度可以直接從後面的層流回到前面的層，完全不衰減：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \frac{\partial H}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \left(1 + \frac{\partial F}{\partial x}\right)
$$

就算 $\frac{\partial F}{\partial x}$ 很小，加上 1 之後梯度也不會消失。

### ResNet34 的完整架構

ResNet34 使用的是 **Basic Block**（對比 ResNet50 以上用的 Bottleneck Block）：

```
Basic Block：
  Conv(3×3, stride=1, padding=1) → BN → ReLU
  Conv(3×3, stride=1, padding=1) → BN
  + skip connection
  → ReLU
```

如果 skip connection 的輸入輸出 channel 數不同（因為 stride=2 的 downsampling），就用 **Projection Shortcut**：

$$
H(x) = F(x) + W_s \cdot x
$$

其中 $W_s$ 是一個 $1 \times 1$ 卷積，用來對齊通道數和空間尺寸。

ResNet34 的完整架構如下：

| 階段 | 操作 | 輸出尺寸 | 通道數 | Block 數 |
|------|------|---------|--------|---------|
| Input | — | 224×224 | 3 | — |
| Stem | Conv(7×7, stride=2) + BN + ReLU | 112×112 | 64 | — |
| Pool | MaxPool(3×3, stride=2) | 56×56 | 64 | — |
| Layer 1 | Basic Block × 3 | 56×56 | 64 | 3 |
| Layer 2 | Basic Block × 4（首個 stride=2） | 28×28 | 128 | 4 |
| Layer 3 | Basic Block × 6（首個 stride=2） | 14×14 | 256 | 6 |
| Layer 4 | Basic Block × 3（首個 stride=2） | 7×7 | 512 | 3 |
| Head | Global Avg Pool + FC | 1×1 | 512 | — |
| Output | Softmax | — | num_classes | — |

共 34 層（1 stem + 3 + 4 + 6 + 3 = 17 個 conv block，每個 block 含 2 層 conv，加上 stem 和 head）。

### Tensor Shape 的完整流動

以輸入 $[B, 224, 224, 3]$ 為例，追蹤 ResNet34 的 tensor 流動：

```
Input:      [B, 224, 224, 3]
Stem Conv:  [B, 112, 112, 64]    <- 7×7, stride 2
MaxPool:    [B, 56, 56, 64]      <- 3×3, stride 2

Layer 1 (×3, 無 downsampling):
  Block 1:  [B, 56, 56, 64]  + skip: [B, 56, 56, 64]   -> [B, 56, 56, 64]
  Block 2:  [B, 56, 56, 64]  + skip: [B, 56, 56, 64]   -> [B, 56, 56, 64]
  Block 3:  [B, 56, 56, 64]  + skip: [B, 56, 56, 64]   -> [B, 56, 56, 64]

Layer 2 (×4, 首個 block stride=2 做 downsampling):
  Block 1:  [B, 28, 28, 128] + skip(projection): [B, 28, 28, 128]
  Block 2~4:[B, 28, 28, 128] + skip(identity)

Layer 3 (×6, 首個 block stride=2):
  Block 1:  [B, 14, 14, 256] + skip(projection): [B, 14, 14, 256]
  Block 2~6:[B, 14, 14, 256] + skip(identity)

Layer 4 (×3, 首個 block stride=2):
  Block 1:  [B, 7, 7, 512]   + skip(projection): [B, 7, 7, 512]
  Block 2~3:[B, 7, 7, 512]   + skip(identity)

Global Avg Pool: [B, 512]
FC:              [B, num_classes]
```

每次 Layer 轉換時，空間尺寸減半、通道數翻倍，這是 CNN 家族的共同慣例：用通道數的增加來補償空間解析度的損失。

### Identity vs. Projection Shortcut

| 情況 | Skip Connection | 計算 |
|------|----------------|------|
| 通道數不變、尺寸不變 | Identity | $H(x) = F(x) + x$ |
| 通道數翻倍、尺寸減半 | Projection | $H(x) = F(x) + W_s x$ |

Projection shortcut 的 $W_s$ 是 $1 \times 1$ 卷積加上 stride=2，既調整通道數，也做空間下採樣。

---

## UNet：讓 CNN 從分類走向分割

### 分割與分類的根本差異

分類只問「這張圖裡有什麼？」，一個標籤就夠了。分割問的是「每個像素屬於哪個類別？」，輸出尺寸要和輸入完全一樣。

這帶來了一個根本矛盾：

> CNN 透過 pooling 和 stride 把空間尺寸一路壓小，但分割需要恢復到原始解析度。

FCN（Fully Convolutional Network）是最早期的解法，用 transposed convolution 把 feature map 上採樣回去，但這樣還原的空間資訊太粗糙，細節完全丟光了。

### UNet 的核心想法

UNet（2015）的設計思路很清楚：

> 下採樣的過程中，每個尺度的特徵都有價值。把這些特徵在上採樣時接回來，讓解碼器可以同時看到「語意（高層抽象）」和「細節（低層局部）」。

這就是 UNet 的 skip connection，但要特別注意，它和 ResNet 的 skip connection **性質完全不同**：

- **ResNet 的 skip connection**：連接同一層的輸入和輸出，目的是緩解梯度消失，讓深層可以學到殘差。
- **UNet 的 skip connection**：連接 encoder 和 decoder 中對應尺度的特徵，目的是把空間細節傳遞給上採樣過程。

兩者都叫 skip connection，但一個是「讓自己學得動」，另一個是「把丟掉的資訊拉回來」。

### UNet 的完整架構

UNet 由三個部分組成：Encoder（壓縮路徑）、Bottleneck、Decoder（擴張路徑）。

**Encoder（壓縮路徑）**

每個 encoder stage 做的事是：Conv → BN → ReLU → Conv → BN → ReLU → MaxPool。前兩個 Conv 提取當前尺度的特徵，MaxPool 把尺寸減半。

```
Input:       [B, 572, 572, 1]      (以原版 UNet 論文尺寸為例)
Encoder 1:   [B, 570, 570, 64]  → MaxPool → [B, 285, 285, 64]
Encoder 2:   [B, 283, 283, 128] → MaxPool → [B, 141, 141, 128]
Encoder 3:   [B, 139, 139, 256] → MaxPool → [B, 69, 69, 256]
Encoder 4:   [B, 67, 67, 512]   → MaxPool → [B, 33, 33, 512]
```

> Note：原版 UNet 用 valid convolution，所以空間尺寸在每個 conv 後都縮小 2。如果改用 same convolution + padding，可以保持尺寸不變，實作更簡單，現代實作多半這樣做。

**Bottleneck**

最底層，空間尺寸最小、語意最高：

```
Bottleneck:  [B, 31, 31, 1024]   (valid) 或 [B, 32, 32, 1024] (same padding)
```

**Decoder（擴張路徑）**

每個 decoder stage：先把 feature map 上採樣（Transposed Conv × 2），然後和 encoder 對應尺度的 feature map 做 **channel-wise concatenate**，再過兩個 Conv。

```
Decoder 4:   UpConv([B, 512]) ++ skip([B, 512]) = [B, 1024] → Conv → [B, 512]
Decoder 3:   UpConv([B, 256]) ++ skip([B, 256]) = [B, 512]  → Conv → [B, 256]
Decoder 2:   UpConv([B, 128]) ++ skip([B, 128]) = [B, 256]  → Conv → [B, 128]
Decoder 1:   UpConv([B, 64])  ++ skip([B, 64])  = [B, 128]  → Conv → [B, 64]
Output:      Conv(1×1) → [B, H, W, num_classes]
```

`++` 代表在通道維度上 concatenate，而不是像 ResNet 那樣做加法（summation）。這是兩者另一個重要的差別：

| | ResNet Skip | UNet Skip |
|--|-------------|-----------|
| 操作 | Element-wise Add | Channel Concatenate |
| 目的 | 梯度流動 + 殘差學習 | 空間細節傳遞 |
| 特徵使用 | 同一路徑的上下游 | 不同路徑的 encoder-decoder |
| 通道數影響 | 不增加 | 翻倍（後面用 Conv 壓回來）|

### Transposed Convolution：上採樣的數學

Transposed Convolution（有時誤稱 deconvolution）是 UNet 做上採樣的核心機制。它的作用是把每個輸入像素展開，乘上 kernel，在輸出空間上疊加。

直觀理解：一般卷積把大輸入對應成小輸出（多對一），Transposed Convolution 把小輸入對應成大輸出（一對多）。

以 stride=2 的 Transposed Convolution 為例，空間尺寸從 $[H, W]$ 擴大到 $[2H, 2W]$。輸出大小公式：

$$
H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + k
$$

另一個常見替代方案是 **Bilinear Upsampling + Conv**：先用雙線性插值把尺寸放大，再接一個普通卷積調整特徵。這種做法更穩定，不容易出現 transposed conv 的棋盤格 artifact。

### UNet 的 Tensor Shape 完整追蹤（Same Padding 版本）

以 $[B, 256, 256, 1]$ 輸入、3 類分割為例：

```
Input:            [B, 256, 256, 1]

=== Encoder ===
E1: 2×Conv(64)    [B, 256, 256, 64]   <- 保存，skip1
    MaxPool       [B, 128, 128, 64]
E2: 2×Conv(128)   [B, 128, 128, 128]  <- 保存，skip2
    MaxPool       [B, 64, 64, 128]
E3: 2×Conv(256)   [B, 64, 64, 256]    <- 保存，skip3
    MaxPool       [B, 32, 32, 256]
E4: 2×Conv(512)   [B, 32, 32, 512]    <- 保存，skip4
    MaxPool       [B, 16, 16, 512]

=== Bottleneck ===
B:  2×Conv(1024)  [B, 16, 16, 1024]

=== Decoder ===
D4: UpConv(512)   [B, 32, 32, 512]
    cat(skip4)    [B, 32, 32, 1024]   <- concatenate
    2×Conv(512)   [B, 32, 32, 512]
D3: UpConv(256)   [B, 64, 64, 256]
    cat(skip3)    [B, 64, 64, 512]
    2×Conv(256)   [B, 64, 64, 256]
D2: UpConv(128)   [B, 128, 128, 128]
    cat(skip2)    [B, 128, 128, 256]
    2×Conv(128)   [B, 128, 128, 128]
D1: UpConv(64)    [B, 256, 256, 64]
    cat(skip1)    [B, 256, 256, 128]
    2×Conv(64)    [B, 256, 256, 64]

Output: Conv(1×1) [B, 256, 256, 3]   <- num_classes = 3
```

### ResNet34 作為 UNet Encoder

UNet 原版的 encoder 是從頭訓練的。但後來發現，用 ImageNet 預訓練的 ResNet34 作為 encoder 效果更好，訓練也更快——這就是 **ResNet34-UNet** 的由來。

設計邏輯非常自然：

- ResNet34 的 Layer 1~4 正好提供了四個尺度的 feature map（56×56, 28×28, 14×14, 7×7）。
- 把這四個 feature map 當成 UNet 的 skip connection 接進 decoder。
- Decoder 負責逐步上採樣，恢復空間解析度。

架構對應關係：

| UNet 部分 | ResNet34 部分 | 尺寸（224×224 輸入）| 通道數 |
|-----------|-------------|---------------------|--------|
| Encoder skip1 | Stem output | 112×112 | 64 |
| Encoder skip2 | Layer 1 output | 56×56 | 64 |
| Encoder skip3 | Layer 2 output | 28×28 | 128 |
| Encoder skip4 | Layer 3 output | 14×14 | 256 |
| Bottleneck | Layer 4 output | 7×7 | 512 |

Decoder 再把 7×7 一路上採樣回 224×224，每一步都接對應的 ResNet34 skip feature。

這個設計的好處是：ResNet34 的預訓練權重已經學到了豐富的 ImageNet 特徵，只需要少量資料就能 fine-tune，不需要從頭訓練整個 encoder，對小資料集（如醫學影像）特別有價值。

---

## 進階任務

CNN 不只做分類。以下幾個方向都是以 CNN 為基礎，再根據任務需求加入不同的設計。

### 物件偵測

**R-CNN 系列**的演化主線是把重複計算消掉。R-CNN 每個候選區域跑一次 CNN，太慢；Fast R-CNN 先跑整張圖，再用 RoI Pooling 從 feature map 截取區域；Faster R-CNN 把候選區域的生成（Region Proposal Network）也學進模型裡，整個流程端到端。

**YOLO** 走的是另一條路：把影像分成格子，每個格子直接預測邊界框和類別，一次 forward pass 完成偵測。犧牲一點精度，換來快很多的速度。

### 語義分割

**FCN** 把分類 CNN 的最後幾個全連接層換成 $1 \times 1$ 卷積，再用 transposed convolution 上採樣，輸出和輸入同尺寸的 class map。

**DeepLabv3+** 結合 atrous convolution 擴大感受野，加上 ASPP（Atrous Spatial Pyramid Pooling）收集多尺度上下文，再用 encoder-decoder 結構恢復解析度。

**Mask R-CNN** 在 Faster R-CNN 基礎上加一條 mask 分支，做到實例分割（每個物件各自一張 mask）。

---

## 常見誤解

**誤解 1：CNN 的 convolution 就是數學上的卷積。**
不是。實作裡幾乎都是 cross-correlation，差別在 kernel 有沒有翻轉。由於 kernel 是訓練出來的，這個差別不影響模型能力，但概念上要清楚。

**誤解 2：kernel 越大，感受野越大，越好。**
不一定。多層 $3 \times 3$ 疊起來的效果等同大 kernel，但參數更少、非線性更多。VGG 就是靠這個思路設計出來的。

**誤解 3：ResNet 和 UNet 的 skip connection 是同一件事。**
不是。ResNet 的 skip connection 做加法，目的是緩解梯度消失、讓深層可以學殘差；UNet 的 skip connection 做 concatenation，目的是把 encoder 的空間細節傳給 decoder。兩者出發點和作用完全不同。

**誤解 4：UNet 只適合醫學影像。**
雖然 UNet 因醫學影像而出名，但只要是需要密集預測（dense prediction）的任務——語義分割、深度估測、光流——UNet 架構都可以用。

**誤解 5：CNN 只能處理影像。**
只要資料是 grid-like topology（音訊波形、時間序列、3D 體積資料），CNN 都能發揮作用。V-Net 就是把 UNet 的卷積全部換成 3D 卷積，用在醫學體積影像分割。

---

## 總結

1. **MLP 之所以不適合影像**，是因為它沒有建入任何關於空間結構的先驗假設，導致參數量爆炸、對平移沒有不變性。CNN 的三個設計原則——局部連接、權重共享、層級組合——正好對應了影像的三個固有性質。

2. **ResNet 解決的是深度退化問題**，而不只是「讓網路更深」。Residual Block 的 skip connection 提供了梯度的直接通道，讓 loss 的梯度可以繞過可能飽和的層直接回傳。ResNet34 用 Basic Block 堆出 34 層，每個 Layer 轉換時通道數翻倍、空間尺寸減半，最後用 Global Average Pooling 代替全連接層。

3. **UNet 解決的是分割任務中「語意 vs. 精度」的矛盾**。Encoder 壓縮空間換取語意，Decoder 恢復解析度，而 skip connection（concatenation）把 encoder 各尺度的細節直接橋接給 decoder，讓上採樣不只是靠瞎猜。用 ResNet34 作為 UNet encoder，則是把預訓練的特徵提取能力引入分割任務的標準做法。

4. **三者之間的關係**：CNN 是基礎；ResNet 修掉了深度 CNN 的訓練問題，讓深層特徵可以真正學起來；UNet 把 CNN encoder 學到的分層特徵，透過 decoder 還原成像素級預測。從分類到分割，架構的演化是對任務需求的直接回應。

---

## Takeaways（關鍵帶走）

- **設計依據資料**：影像的局部性與平移結構決定了 CNN 的 inductive bias，是模型能有效學習的關鍵。
- **小 kernel 多層勝過大 kernel**：堆疊多個 $3\times3$ 卷積可以在更低參數下獲得更大感受野與更多非線性。
- **殘差與 skip 的角色分明**：ResNet 的 skip 解決優化問題；UNet 的 skip 幫助空間細節復原。
- **實務建議**：資料有限時，使用 ImageNet 預訓練的 encoder（如 ResNet）接 UNet decoder，能顯著提升分割任務的泛化表現。
