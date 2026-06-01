---
type: Blog-post
tags:
  - Deep-Learning
  - note
  - Generative-Model
created: 2026-05-28
updated: 2026-05-28
authors: WeiTa
status: active
---


> 上一篇的 VAE 已經能生成新樣本，但生成品質有限。
> 這篇要回答的問題是：有沒有更好的路？
> Normalizing Flow 和 Diffusion 給出了兩個截然不同的答案，但它們出發的問題是同一個。

## 這篇文章想回答什麼問題？

VAE問世之後，有個問題是: **生成的圖有點模糊**。
這不是偶然——它源自 VAE 設計上一個無法迴避的取捨：

VAE 用一個 encoder 去**近似** posterior，再用 ELBO 代替真正的 log-likelihood 來訓練。
這個近似讓訓練變得可行，但也引入了一道玻璃牆：模型訓練永遠只是在優化一個下界，不是真正的 likelihood。

有沒有辦法**精確**算出 $\log p_\theta(x)$，而不需要近似？

這篇文章介紹兩條試圖打破這道牆的路線：

1. **Normalizing Flow**：設計一個特殊的可逆函數，讓 $p_\theta(x)$ 可以透過變數變換定理精確計算。
2. **Diffusion Model**：放棄「精確 likelihood」的目標，改用一種更聰明的訓練代理——把生成問題拆成無數個簡單的去噪步驟。

如果把兩條路的哲學濃縮成一句話：

> **Flow 說**：「我要找一條可逆的路，在資料和噪聲之間精確地雙向通行。」
> **Diffusion 說**：「我不需要一條精確的路。我把資料慢慢弄髒，再學會一步步修復它。」

這篇文章的閱讀路線：
- 先從 VAE 的限制出發，理解我們真正想要的是什麼
- 再看 Flow 如何用變數變換解決密度計算
- 最後看 Diffusion 如何用一條截然不同的思路，得到目前最好的生成品質。

---

## 起點：VAE 留下了什麼問題？

在進入 Flow 之前，先把「我們在解什麼問題」說清楚。

VAE 的訓練目標是最大化 ELBO：

$$
\mathcal{L}_\text{ELBO}(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建好}} - \underbrace{KL(q_\phi(z|x) \| p(z))}_{\text{latent 不要太怪}}
$$

這個目標和真正的 log-likelihood 之間有一個 gap：

$$
\log p_\theta(x) = \mathcal{L}_\text{ELBO}(x) + \underbrace{KL(q_\phi(z|x) \| p_\theta(z|x))}_{\ge 0}
$$

這個 KL 項永遠 $\ge 0$，所以 ELBO 永遠只是下界。
當 encoder 的近似越精確，這個 gap 越小；但在 nonlinear decoder 下，posterior $p_\theta(z|x)$ 永遠無法被完美近似，所以 gap 永遠存在。

這帶來一個自然的問題：

> 有沒有辦法讓 $p_\theta(x)$ **直接可算**，跳過 approximate posterior？

Normalizing Flow 的答案是：**有，但你需要讓映射具有特殊結構。**

---

## 密度模型需要同時做兩件事

在討論 Flow 之前，先把「一個好的密度模型需要做什麼」的要求說清楚，因為這直接決定了 Flow 的架構必須長什麼樣子。

假設資料 $x \in \mathbb{R}^n$ 來自某個未知分佈 $p_\text{data}(x)$。
我們想訓練一個模型 $p_\theta(x)$，讓它：

1. **Density evaluation**：給定一個新樣本 $x$，能計算出 $p_\theta(x)$。（用於訓練、異常偵測）
2. **Sampling**：能從 $p_\theta(x)$ 中抽出有意義的新樣本。（用於生成）

這兩件事看起來簡單，但對高維資料（影像、音訊）來說，同時做到非常困難。

### 最簡單的模型-GMM為什麼不夠？

最簡單的密度模型是 Gaussian Mixture Model：

$$
p_\theta(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x;\, \mu_k,\, \Sigma_k)
$$

GMM 可以做 density evaluation（直接代入公式），也可以做 sampling（先抽 component，再從 Gaussian 抽樣）。

理論上只要 component 數夠多，它可以逼近任意分佈。

**但實務上，GMM 對高維結構化資料有根本限制：**

真實影像不是均勻分佈在整個 $\mathbb{R}^n$ 裡——256×256 的 RGB 影像有 196,608 個像素，但「看起來像真實照片」的點在這個空間裡極度稀疏，集中在一個低維的**流形（Manifold）**上。

GMM 的每個 component 是一個「各方向都有厚度的橢球」，它能告訴你「哪一帶有資料」，但無法捕捉這種細長、彎曲的流形結構。
從這樣的模型裡抽出來的樣本，往往是「像素隨機擾動後的模糊影像」，落在流形附近但不在流形上。

> **直覺**：想像把**所有**真實人臉的影像投影到高維空間，它們形成一個複雜的低維曲面。
> GMM 能「大致圈出」這個曲面所在的區域，但無法描述曲面的精確形狀。
> Flow 和 Diffusion 的目標，就是學到這個形狀。

---

## Normalizing Flow：用可逆映射精確算密度

### 核心想法：把複雜分佈變換成簡單分佈

Flow 的出發點是一個非常直接的想法：

如果資料 $x$ 的分佈很複雜，那麼找一個函數 $f_\theta$ 把 $x$ 轉換成 $z$，讓 $z$ 的分佈變得簡單（例如標準常態 $\mathcal{N}(0, I)$），不就好了嗎？

$$
z = f_\theta(x), \qquad z \sim \mathcal{N}(0, I)
$$

這個想法和 VAE 的 encoder 很像——但 VAE 的 encoder 輸出的是「$z$ 的分佈參數」（mean 和 variance），本質上是在做**近似推論**；Flow 要求更高：$f_\theta$ 必須是**可逆**的，讓我們能：

- 從 $x$ 算 $z$（編碼方向）
- 從 $z$ 還原 $x = f_\theta^{-1}(z)$（採樣方向）

**可逆性的代價**：因為 $x$ 和 $z$ 維度必須相同，且映射是一對一的，所以 Flow 沒有「壓縮」——它是在學一個彎曲的坐標系，而不是在學一個低維瓶頸。

### 為什麼可以計算密度？變數變換定理

可逆映射讓我們可以用**變數變換定理**精確計算 $p_\theta(x)$。

先用一維的情況建立直覺：
假設 $z = f_\theta(x)$，且 $f_\theta$ 可逆可微。

直覺上，「*機率質量*(Probability Mass)必須守恆」——  $x$ 附近 $|dx|$ 這一段的機率質量，等於 $z$ 附近 $|dz|$ 的機率質量：

$$
p_\theta(x)\,|dx| = p_Z(z)\,|dz|
$$

由於 $|dz| = |f'_\theta(x)|\,|dx|$（導數就是局部的「拉伸比例」），代入整理：

$$
\boxed{p_\theta(x) = p_Z(f_\theta(x)) \cdot \left|\frac{df_\theta(x)}{dx}\right|}
$$

**這個式子在說什麼**：$p_\theta(x)$ 由兩部分決定——
- $p_Z(f_\theta(x))$：$x$ 被映射到 $z$ 之後，$z$ 在標準常態下有多大的機率密度（「落在哪裡」）。
- $\left|\frac{df_\theta}{dx}\right|$：映射在 $x$ 附近把空間拉伸了多少倍（「空間被壓縮還是拉伸」）。

**角色類比**：把 $f_\theta$ 想像成一個橡皮筋——你把資料空間的橡皮筋拉伸成均勻的標準常態形狀。


某個點的密度，取決於「那個點在標準常態裡落在哪裡」乘以「橡皮筋在那個點被拉伸了多少」。

把這個代入最大似然目標：

$$
\max_\theta \sum_i \left[\underbrace{\log p_Z(f_\theta(x^{(i)}))}_{\text{映射到 z 的高機率區域}} + \underbrace{\log \left|\frac{df_\theta}{dx}(x^{(i)})\right|}_{\text{映射的拉伸合理}}\right]
$$

訓練同時在做兩件事：
**讓每個資料點 $x$ 被映射到 $z$ 空間的高機率區域，同時讓映射本身不要過度扭曲空間。**

### 高維推廣：Jacobian 行列式的計算困難

在高維 $x \in \mathbb{R}^n$，一維的導數變成 Jacobian 矩陣 $\frac{\partial f_\theta(x)}{\partial x} \in \mathbb{R}^{n \times n}$，訓練目標變成：

$$
\log p_\theta(x) = \log p_Z(f_\theta(x)) + \log\left|\det \frac{\partial f_\theta(x)}{\partial x}\right|
$$

**這裡遇到了一個工程瓶頸**：
一般的 $n \times n$ 矩陣計算 determinant 的複雜度是 $O(n^3)$。
對 256×256 的影像（$n \approx 196{,}608$），這根本無法計算。

這就是 Flow 架構設計的核心問題：

> **如何設計 $f_\theta$，讓它既有足夠的表達力，又讓 Jacobian determinant 容易計算？**

### 多層串聯：複雜度從組合中來

單層映射的表達力有限。
Flow 的解法是把多個簡單的可逆 block 串起來：

$$
z = f_k \circ \cdots \circ f_2 \circ f_1(x)
$$

由*鏈式法則* (chain law)，整體 Jacobian 是各層 Jacobian 的乘積；而 log determinant 對乘積可以分解成加總：

$$
\log\left|\det \frac{\partial z}{\partial x}\right| = \sum_{i=1}^{k}\log\left|\det \frac{\partial f_i}{\partial f_{i-1}}\right|
$$

**這個分解的意義**：複雜的密度模型 = 一串簡單可逆變換的組合，每層只需要算自己的 Jacobian 貢獻，不需要看整條鏈。訓練時每一層可以獨立計算梯度。

---

## Flow 的架構設計：讓 Jacobian 變得容易算

### Affine Coupling Layer（RealNVP 的核心）

Affine coupling layer 是目前最廣泛使用的 Flow block，它用一個巧妙的設計讓 Jacobian determinant 從 $O(n^3)$ 降到 $O(n)$。

**設計思路**：把輸入維度 split 成兩半 $x = (x_a, x_b)$，前半部**完全不動**，後半部做一個以前半部為條件的仿射變換：

$$
z_a = x_a
$$

$$
z_b = x_b \odot s_\theta(x_a) + t_\theta(x_a)
$$

其中 $s_\theta$（scale）和 $t_\theta$（translation）可以是任意複雜的神經網路。

**為什麼可逆**：因為 $z_a = x_a$ 直接給了你 $x_a$，所以可以還原 $x_b = (z_b - t_\theta(z_a)) / s_\theta(z_a)$。$s_\theta$ 和 $t_\theta$ 不需要是可逆的——它們只是用來定義變換，不是被取反。

**為什麼 Jacobian 容易算**：這個設計讓 Jacobian 矩陣變成**下三角**形：

$$
\frac{\partial z}{\partial x}
=
\begin{bmatrix}
I & 0 \\
\frac{\partial z_b}{\partial x_a} & \mathrm{diag}(s_\theta(x_a))
\end{bmatrix}
$$

下三角矩陣的 determinant 等於對角線元素的乘積——這讓計算複雜度從 $O(n^3)$ 降到 $O(n)$：

$$
\log\left|\det \frac{\partial z}{\partial x}\right| = \sum_j \log |s_\theta(x_a)_j|
$$

**這個式子的角色**：它是訓練 loss 的一部分（Jacobian 項），告訴模型「這個 scale 合不合理」。如果 $s_\theta$ 把某個維度壓縮得太厲害，這項會有懲罰。

> **Insight**：Affine coupling 把「可逆性」和「表達力」解耦——前半部不動保證可逆，後半部的複雜變換由任意神經網路控制。
> 代價是：因為前半部永遠不動，需要交替改變「哪半部被變換、哪半部保持不動」，確保所有維度都能被學到。這就是 RealNVP 的核心。

### Autoregressive Flow：速度的取捨

另一類常見的 Flow 是基於 autoregressive factorization。把聯合分佈按 chain rule 逐維分解：

$$
p(y_1,y_2,\dots,y_n) = p(y_1)\prod_{i=2}^{n} p(y_i \mid y_1,\dots,y_{i-1})
$$

如果每個條件分佈都是 Gaussian，對應的 flow 把資料映射到標準常態的方式就是：

$$
z_i = \frac{y_i - \mu_i(y_{<i})}{\sigma_i(y_{<i})}
$$

這樣的 Jacobian 天然是下三角矩陣。但這個結構有一個不可迴避的速度 trade-off：

| | Autoregressive Flow (AF) | Inverse Autoregressive Flow (IAF) |
|--|--|--|
| Density evaluation | 快（$O(n)$ 平行） | 慢（必須序列計算） |
| Sampling | 慢（必須序列生成） | 快（$O(n)$ 平行） |

**這個 trade-off 的來源**：AF 從 $y \to z$ 方向天然可以平行（每個 $z_i$ 只依賴已知的 $y_{<i}$），但反向 $z \to y$ 必須序列進行（因為生成 $y_i$ 需要已知 $y_{<i}$）。
IAF 把計算方向反過來，採樣變快但 likelihood 計算變慢。

---

## 實作細節：離散影像需要 Dequantization

Flow 是為連續密度設計的，但影像資料是 8-bit 整數（每個 pixel 值是 $0 \sim 255$）。如果直接把離散整數點丟進連續密度模型，模型會把所有機率質量集中到這些點上，density 趨近無窮大——這是數學意義上的退化（degeneracy）。

**解法**：在訓練時給離散資料加入均勻噪聲：

$$
u \sim \mathrm{Uniform}([0,1)^D), \qquad \tilde{x} = x + u
$$

**為什麼這樣合理**：真實的離散 likelihood 目標等價於：

$$
\mathbb{E}_{x \sim P_\text{data}}\!\left[\log \int_{[0,1)^D} p_\text{model}(x+u)\,du\right]
\ge \mathbb{E}_{x \sim P_\text{data}}\!\left[\mathbb{E}_{u}[\log p_\text{model}(x+u)]\right]
$$

不等式來自 Jensen inequality（$\log$ 是凹函數）。這說明：**在加了噪聲的連續樣本上最大化 log-likelihood，等價於在優化離散 likelihood 的下界**。

> **Warning**：Dequantization 是 Flow 能用在影像上的必要前處理。跳過它的話，likelihood 數字可能看起來很高，但實際上是 degeneracy 的假象。

---

## Flow 的總結：精確的代價

Flow 給了我們精確計算 $\log p_\theta(x)$ 的能力——這是 VAE 做不到的。但精確性有代價：

- **維度限制**：$x$ 和 $z$ 必須同維度，不能壓縮。模型無法學到低維表示。
- **架構約束**：為了讓 Jacobian determinant 可算，架構必須滿足特殊結構（下三角 Jacobian），這限制了模型的表達力。
- **影像品質的天花板**：相比 GAN 和 Diffusion，Flow 的生成品質偏低。精確的 likelihood 和高品質的樣本，在實務上並不等價。

這讓我們問：有沒有方法在放棄「精確 likelihood」的前提下，得到更高品質的生成？

Diffusion 的答案是：**有，而且這條路的思路和 Flow 完全不同。**

---

## Diffusion Model：換一種思路

### 從 Flow 的硬性限制出發

Flow 的核心限制是：$f_\theta$ 必須可逆，且 Jacobian determinant 必須容易算。這兩個要求相互制約，讓架構設計非常受限。

Diffusion 放棄了「找一個可逆映射」的目標。它的出發點是一個完全不同的觀察：

> **如果把資料一步一步加噪，最終會變成純噪聲；那麼如果學會了如何「一步一步去噪」，就能從噪聲反向走回資料。**

這個想法的精妙之處在於：每一步去噪的目標都很簡單（只需要「稍微去掉一點噪聲」），而複雜的生成過程是由很多個簡單步驟組合而成的。

**比喻**：雕塑家不會一刀就刻出大衛像——他會一點一點地鑿，每一刀只改動一小部分。Diffusion 就是把「從噪聲生成影像」這件一步很難的事，拆成一千刀，每刀都很容易學。

### Diffusion 是一種 Latent Variable Model

從 VAE 的角度來看，Diffusion 也可以被理解為一種 latent variable model。VAE 引入一個隱變數 $z$ 描述生成：

$$
z \to x
$$

Diffusion 引入的是一整條隱變數鏈：

$$
x_0 \xrightarrow{\text{加噪}} x_1 \xrightarrow{\text{加噪}} x_2 \to \cdots \to x_T
$$

其中 $x_0$ 是真實資料，$x_1, x_2, \dots, x_T$ 都是 latent variables。和 VAE 的差別：

| | VAE | Diffusion |
|--|--|--|
| Latent 維度 | 通常遠低於 $x$ | 和 $x$ 同維度 |
| Encoder | 可訓練，輸出分佈參數 | **固定不訓練**，是事先設計好的加噪程序 |
| Decoder | 可訓練 | 可訓練（去噪網路）|
| Latent 層數 | 1 層 | $T$ 層（通常 $T = 1000$）|
| 訓練目標 | ELBO（一項 KL）| ELBO（$T$ 項 KL 的加總）|

**Encoder 不需要訓練**是 Diffusion 最重要的設計選擇：既然加噪的方式是固定的（由人設計），就不需要讓模型去學，整個學習問題聚焦在「如何去噪」上。

---

## Forward Process：把資料慢慢弄髒

Forward process 是一個**固定的、不需要訓練的** Markov chain，定義了如何把資料逐步加噪：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I\right)
$$

**這個式子在做什麼**：每一步把前一步的訊號縮小（乘 $\sqrt{1-\beta_t}$），再疊上一個 Gaussian 噪聲（方差 $\beta_t$）。$\beta_t \in (0,1)$ 是 noise schedule，是人工設計的超參數。

**為什麼每步只加一點點噪聲？**

這不只是工程選擇，而是讓 Diffusion 能訓練的數學前提：當每步的 $\beta_t$ 足夠小時，反向去噪的每一步可以被近似為 Gaussian 分佈，模型才能用參數化的 Gaussian 來學習。如果每步加太多噪聲，反向步驟就需要「一大步跳回去」，這幾乎無法用 Gaussian 近似，也就無法學習。

**累積公式：直接從 $x_0$ 跳到任意 $x_t$**

定義 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$，多步加噪可以寫成 closed form：

$$
\boxed{x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0,I)}
$$

**這個公式的意義**：不需要一步一步算，可以直接從原始資料 $x_0$ 採樣任意時間步的 $x_t$。$\sqrt{\bar{\alpha}_t}$ 是剩餘 signal 的強度，$\sqrt{1-\bar{\alpha}_t}$ 是累積 noise 的強度——兩者平方和等於 1（保持能量守恆）。

- 當 $t = 0$：$x_0 = x_0$，完全是原始資料。
- 當 $t \to T$：$\bar{\alpha}_T \approx 0$，$x_T \approx \epsilon \sim \mathcal{N}(0, I)$，幾乎是純噪聲。

> **Note**：這個 closed form 之所以存在，是因為多個獨立 Gaussian 的線性組合仍然是 Gaussian。這讓訓練時不需要逐步執行 forward process——可以直接從 $(x_0, t, \epsilon)$ 計算 $x_t$，大幅提升訓練效率。

---

## Reverse Process：學會去噪

生成時，從純噪聲 $x_T$ 出發，執行反向鏈：

$$
x_T \to x_{T-1} \to \cdots \to x_1 \to x_0
$$

每一步的反向條件分佈 $p_\theta(x_{t-1} \mid x_t)$ 是模型要學的東西。DDPM 把它參數化為 Gaussian：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\right)
$$

**為什麼可以假設 Gaussian**：當 $\beta_t$ 足夠小，從加噪過程的理論可以證明，逆向分佈在局部近似為 Gaussian。這就是為什麼 Diffusion 要把噪聲加得很小、很緩慢——它在為反向的 Gaussian 假設製造成立的條件。

---

## 訓練目標：從 ELBO 到預測噪聲

### 為什麼不能直接最大化 likelihood？

和 VAE 一樣，$p_\theta(x_0)$ 需要對整條 latent 鏈積分：

$$
p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}
$$

$T$ 層積分完全無法直接計算。

### ELBO 的推導

引入 forward process $q(x_{1:T} \mid x_0)$ 作為近似後驗（和 VAE 的 encoder 扮演的角色類似），用 Jensen 不等式得到 ELBO：

$$
\log p_\theta(x_0) \ge \mathbb{E}_{q(x_{1:T}|x_0)}\!\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] =: \mathcal{L}
$$

展開後，ELBO 整理成三項：

$$
\mathcal{L}
= \underbrace{\mathbb{E}_q[\log p_\theta(x_0 \mid x_1)]}_{\text{最終重建項}}
- \underbrace{KL(q(x_T|x_0) \| p(x_T))}_{\text{先驗匹配項（近乎常數）}}
- \sum_{t=2}^{T} \underbrace{KL\!\left(q(x_{t-1}|x_t,x_0) \,\|\, p_\theta(x_{t-1}|x_t)\right)}_{\mathcal{L}_{t-1}:\text{去噪匹配項（訓練主體）}}
$$

**三項各自的角色**：

- **最終重建項**：把接近原始影像的 $x_1$ 還原成 $x_0$，類似 VAE 的重建 loss，但只是一小步。
- **先驗匹配項**：要求 $x_T$ 足夠接近標準 Gaussian。因為 forward process 設計時 $\bar{\alpha}_T \approx 0$，這項自動滿足，**訓練時通常忽略**。
- **去噪匹配項**：這是訓練的主體。對每個時間步 $t$，要求模型的去噪步驟 $p_\theta(x_{t-1}|x_t)$ 盡量吻合「知道 $x_0$ 和 $x_t$ 時的真正後驗」$q(x_{t-1}|x_t,x_0)$。

**為什麼去噪匹配項有 Closed Form？**

$q(x_{t-1}|x_t,x_0)$ 可以用 Bayes rule 推出，因為所有分佈都是 Gaussian，結果仍然是 Gaussian：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I)
$$

其中：

$$
\tilde{\mu}_t(x_t, x_0)
= \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}x_0
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t, \qquad
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
$$

**這個式子在說什麼**：$\tilde{\mu}_t$ 是 $x_0$ 和 $x_t$ 的加權平均——「用原始資料告訴你應該去噪到哪裡，用當前帶噪版本告訴你現在在哪裡」。$\tilde{\beta}_t$ 是這一步去噪後剩餘的不確定性。

而模型 $p_\theta$ 也被設成 Gaussian，所以 $\mathcal{L}_{t-1}$ 就是兩個 Gaussian 之間的 KL，有 closed form：

$$
\mathcal{L}_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2\right] + C
$$

**這是訓練的核心**：$\mathcal{L}_{t-1}$ 在說「模型預測的去噪 mean，要和真正的後驗 mean 盡量接近」。

### Reparameterization：為什麼預測噪聲？

到這裡，訓練目標仍然是讓 $\mu_\theta(x_t, t)$ 逼近 $\tilde{\mu}_t(x_t, x_0)$。

但注意 forward process 的 closed form：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，可以反解出：

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon\right)
$$

代入 $\tilde{\mu}_t$ 的公式，整理後可以把目標改寫成：

$$
\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)
$$

這說明，讓 $\mu_\theta(x_t, t)$ 逼近 $\tilde{\mu}_t$，等價於讓網路去預測噪聲 $\epsilon$：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)
$$

代入 $\mathcal{L}_{t-1}$，訓練目標化簡為純噪聲回歸：

$$
\boxed{\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\; t\right)\|^2\right]}
$$

**為什麼預測 $\epsilon$ 比直接預測 $\mu_\theta$ 更好？**

- **目標標準化**：$\epsilon \sim \mathcal{N}(0, I)$ 是固定的，每個時間步 $t$ 的學習目標都在同一個尺度。如果預測 $x_0$，目標值域會隨 $t$ 變化（$t$ 大時訊號弱，很難預測），訓練不穩定。
- **Residual learning 的精神**：學「加進去了什麼噪聲」（偏差），比學「原始影像是什麼」（絕對值）更容易——這和 ResNet 學殘差的邏輯相同。

> **Note**：理論上 $\mathcal{L}_{t-1}$ 前面還有一個時間步相關的係數 $w_t$，DDPM 論文發現**去掉這個係數**（也就是用 $\mathcal{L}_\text{simple}$）實際效果更好。這是理論形式和最佳化實務之間常見的落差：有時候正確的理論目標反而不是最好的優化目標。

---

## 訓練與採樣流程

### 訓練

```
1. 從資料集取 x_0
2. 均勻抽時間步  t ~ Uniform{1, ..., T}
3. 抽噪聲       ε ~ N(0, I)，同 x_0 的維度
4. 直接算       x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε    ← closed form，不需要跑迴圈
5. 前向推論      ε_pred = ε_θ(x_t, t)
6. 計算 loss    ||ε - ε_pred||²  並更新參數
```

**訓練的關鍵**：因為 forward process 有 closed form，每個訓練步驟只需要**隨機抽一個時間步 $t$**，直接計算對應的 $x_t$，再做一次前向 + 反向傳播。不需要跑完整條加噪序列，也不需要 BPTT。訓練效率和 VAE 差不多，這是 Diffusion 能規模化的原因。

### 採樣（Inference）

```
1. 從純噪聲出發  x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
       用 ε_θ(x_t, t) 計算 μ_θ(x_t, t)
       x_{t-1} = μ_θ(x_t, t) + σ_t · η,   η ~ N(0, I)    ← 加一點隨機性，保持多樣性
3. 輸出 x_0
```

採樣需要執行 $T$ 步（通常 $T = 1000$），每步都要跑一次神經網路前向推論。這是 Diffusion 最大的缺點——相比 VAE 和 GAN 的單步生成，Diffusion 慢了幾個數量級。

> **Note**：後續的 DDIM（Denoising Diffusion Implicit Models）把 Markov chain 改為 deterministic ODE，可以在只用 50 步的情況下生成接近品質的樣本，大幅改善採樣速度。

---


## Diffusion 的條件控制：Conditional DDPM

上一篇我們學了 DDPM 的基本框架：固定加噪 + 學習去噪，訓練目標是 $\mathcal{L}_\text{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$。

這篇要解決的問題是：**如何讓生成器聽指令？**

無條件 DDPM 可以生成多樣的樣本，但你無法控制「生成什麼」。實際應用幾乎都需要條件輸入——你要生成「一隻坐著的橘貓」，而不是「任意一種動物」。

### 方法一：Classifier Guidance（分類器引導）

**想法**：在已有訓練好的無條件 Diffusion model 的前提下，用一個額外的分類器 $p_\phi(c \mid x_t)$ 來引導採樣方向。

由 Bayes 定理，條件分佈的 score（對數密度的梯度）可以分解為：

$$
\nabla_{x_t} \log p(x_t \mid c) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{無條件 score}} + \underbrace{\nabla_{x_t} \log p_\phi(c \mid x_t)}_{\text{分類器的引導}}
$$

**這個式子在說什麼**：採樣方向 = 「無條件地往高密度移動」 + 「往使條件 $c$ 更可能的方向移動」。加入分類器梯度，就是在採樣時把生成結果推向符合條件的區域。

採樣時把這個修正後的 score 用於更新 $x_t$：

$$
\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t) - \sqrt{1-\bar{\alpha}_t}\, s \cdot \nabla_{x_t} \log p_\phi(c \mid x_t)
$$

其中 $s > 1$ 是 guidance scale，越大則越向 condition 靠攏，但多樣性也會下降。

**缺點**：需要額外訓練一個能處理帶噪影像 $x_t$ 的分類器 $p_\phi(c \mid x_t)$（和普通分類器不同，它需要對所有噪聲程度都能工作）。

### 方法二：Classifier-Free Guidance（無分類器引導）

Ho & Salimans（2022）提出一個更優雅的方法：不需要額外的分類器，只需要訓練一個**同時支援有條件和無條件生成**的 Diffusion model。

**訓練方式**：訓練時以一定機率（例如 10~20%）把條件 $c$ 替換成空條件 $\emptyset$（null token），讓同一個 $\epsilon_\theta$ 同時學習：

$$
\epsilon_\theta(x_t, t, c) \quad \text{（有條件）} \qquad \text{和} \qquad \epsilon_\theta(x_t, t, \emptyset) \quad \text{（無條件）}
$$

**採樣時**，把兩者線性組合，用 guidance scale $w$ 控制條件強度：

$$
\boxed{\hat{\epsilon} = (1+w)\,\epsilon_\theta(x_t, t, c) - w\,\epsilon_\theta(x_t, t, \emptyset)}
$$

**這個公式在說什麼**：改寫為 $\hat{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + (1+w)\underbrace{[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)]}_{\text{條件和無條件的方向差}}$。採樣方向 = 無條件方向 + $(1+w)$ 倍的「條件比無條件多出的方向」。

- $w = 0$：退化為無條件生成。
- $w$ 越大：生成結果越符合條件 $c$，但多樣性越低（過度對齊 condition）。
- 實務上 $w \approx 1 \sim 7.5$ 是常見範圍（依任務和模型而定）。

> **Insight**：Classifier-free guidance 的漂亮之處在於不需要額外模型。同一個 $\epsilon_\theta$ 在訓練時「有時看條件、有時不看條件」，就同時學會了兩種能力。Stable Diffusion、DALL-E 2、Imagen 全都採用這個策略。

### Condition 如何注入網路？

Classifier-free guidance 告訴我們「採樣時怎麼用 condition」，但訓練時 condition $c$ 要如何傳入去噪網路 $\epsilon_\theta$？

以 DDPM 最常用的 U-Net 架構為例，有三種主要注入方式：

**時間步 $t$（每個模型都需要）**：用 sinusoidal positional encoding 把 $t$ 編碼成向量，接 MLP 後得到 time embedding，透過 **Adaptive Group Normalization（AdaGN）** 注入每個 residual block：

$$
h \leftarrow (1 + \gamma_t) \cdot \text{GroupNorm}(h) + \delta_t
$$

其中 $\gamma_t, \delta_t$ 從 time embedding 線性投影而來。

**類別標籤**：把 class label 做 embedding，同樣透過 AdaGN 注入（和時間步類似，只是來源不同）。

**文字提示（Text Prompt）**：用預訓練的文字 encoder（如 CLIP 或 T5）把文字轉成 token sequence，透過 **Cross-Attention** 讓 U-Net 的每個空間位置動態參考文字資訊：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中 $Q$ 來自影像特徵（當前 layer 的 feature map），$K, V$ 來自文字 embedding。

**這個機制在說什麼**：每個影像 patch 在更新自己的特徵時，都能「查詢」文字 embedding，動態決定要關注哪個文字 token。這讓生成過程能細粒度地把文字描述和影像位置對應起來。

### Conditional DDPM 的完整流程

以「文字引導影像生成」為例，整理訓練和採樣的完整流程。

**訓練流程：**

```
輸入：(x_0, c)  # 例如：一張貓的影像 + 文字 "a cat sitting on a chair"

1. 均勻抽 t ~ Uniform{1, ..., T}
2. 抽噪聲 ε ~ N(0, I)，和 x_0 同維度
3. 計算加噪後的 x_t = √ᾱₜ · x_0 + √(1-ᾱₜ) · ε     ← closed form，直接計算
4. 以機率 p_uncond（例如 0.1）把 c 替換成 ∅           ← 讓模型同時學有條件和無條件
5. 前向推論：ε_pred = ε_θ(x_t, t, c)
6. Loss = ||ε - ε_pred||²
7. 反向傳播，更新 θ
```

**採樣流程（Classifier-Free Guidance）：**

```
輸入：條件 c，guidance scale w

1. 抽 x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. ε_cond   = ε_θ(xₜ, t, c)        # 有條件推論
   b. ε_uncond = ε_θ(xₜ, t, ∅)        # 無條件推論
   c. ε_guided = (1+w)·ε_cond - w·ε_uncond   # 線性組合
   d. 計算 μ_θ（用 ε_guided 和 xₜ）
   e. 若 t > 1：xₜ₋₁ ~ N(μ_θ, σₜ²I)
      若 t = 1：x₀ = μ_θ
3. 輸出 x_0
```

注意採樣時每步需要**兩次前向推論**（有條件和無條件各一次），這讓 Conditional DDPM 的採樣比無條件更慢兩倍。

---

## 生成模型路線圖

把目前系列討論過的生成模型放在一起，從「用什麼方式學 $p_\theta(x)$」的角度比較：

| | VAE | Normalizing Flow | Diffusion |
|--|--|--|--|
| 訓練目標 | ELBO（近似下界）| 精確 MLE | ELBO（近似下界）|
| Likelihood 可算 | 近似（ELBO） | **精確** | 近似（ELBO）|
| Sampling 速度 | 快（1 步）| 快（1 步）| 慢（$T$ 步）|
| 生成品質 | 偏平滑 | 中等 | **高（SOTA）** |
| 訓練穩定性 | 穩定 | 穩定 | 穩定 |
| 架構約束 | Encoder-Decoder | **可逆映射（嚴格限制）**| 去噪網路（無特殊要求）|
| Latent 維度 | 低維壓縮 | 同維度（無壓縮）| 同維度（多層）|

沒有一個模型在所有維度都最好。下一篇（第三部曲）會引入 GAN，它走了一條和這三個完全不同的路線：不問 likelihood，直接用對抗博弈學分佈。

---

## 常見誤解

**誤解 1：Flow 和 VAE 都是把資料壓進 latent，它們很類似。**

不是。VAE 的 encoder 輸出**分佈參數**（mean 和 variance），mapping 是 stochastic 的，$z$ 的維度遠低於 $x$。Flow 的映射是**確定性的且嚴格可逆的**，$x$ 和 $z$ 維度相同，不存在任何壓縮或近似。核心差別：Flow 可以**精確**算 likelihood，VAE 只能算下界。

**誤解 2：Diffusion 訓練時要跑完整的加噪序列再反傳梯度。**

不需要。因為 forward process 有 closed form $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，可以直接從 $(x_0, t, \epsilon)$ 計算任意時間步的 $x_t$。每個訓練步驟只需要抽一個 $t$，計算量和 VAE 差不多。

**誤解 3：Diffusion 和 Flow 都是多步的，所以思路相似。**

完全不同。Flow 的多步是把多個可逆映射串聯，整體仍然是一個**確定性的雙射**；Diffusion 的多步是一條 **Markov chain**，每步都是隨機轉移，forward process 甚至是固定不訓練的。

**誤解 4：Diffusion 的採樣速度很慢，所以它不實用。**

DDIM 等方法把採樣步數從 1000 步壓縮到 50 步以下，品質損失很小。Stable Diffusion、DALL-E 2、Imagen 都是基於 Diffusion，且是目前最廣泛使用的圖像生成框架。慢的採樣速度是可以工程優化的問題。

---

## 總結

1. **VAE 的限制是訓練目標的近似性**——ELBO 只是 $\log p_\theta(x)$ 的下界，這個 gap 限制了生成品質的上限。

2. **Normalizing Flow** 用可逆映射 + 變數變換定理，讓 $\log p_\theta(x)$ 精確可算。Affine coupling layer 把 Jacobian 變成下三角矩陣，把複雜度從 $O(n^3)$ 降到 $O(n)$。代價是架構嚴格受限於可逆性，且無法學低維表示。

3. **Diffusion** 放棄精確 likelihood 的目標，改用固定加噪 + 學習去噪的策略。Forward process 的 closed form 讓訓練非常高效；ELBO 推導後，訓練目標化簡為預測加入的噪聲 $\epsilon$，直觀且穩定。

4. **預測噪聲（而非預測 $x_0$ 或 $\mu_\theta$）** 是讓 Diffusion 訓練穩定的關鍵工程選擇：目標尺度統一、類似 residual learning 的精神。

5. **生成模型沒有最好，只有最合適**：Flow 適合需要精確 likelihood 的場景；Diffusion 目前生成品質最高，但採樣慢；VAE 速度最快但品質有限。下一篇的 GAN 則走了完全不同的路線——不算 likelihood，靠對抗博弈來學習分佈。
