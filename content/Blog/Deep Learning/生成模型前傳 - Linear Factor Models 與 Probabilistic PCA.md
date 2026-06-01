---
type: Blog-post
tags:
  - Deep-Learning
  - note
created: 2026-05-27
updated: 2026-05-30
authors: WeiTa
status: active
---

> 這篇文章要說的不只是「把 PCA 加上機率」，而是回答一個更根本的問題：
> 如果資料背後有我們看不見的解釋因素在驅動，要怎麼讓模型把它們學出來？

## 這篇文章想回答什麼問題？

在讀這篇之前，你可能已經學過幾種能把高維資料壓縮的方法：
PCA 找主方向、Autoencoder（自動編碼器）學非線性壓縮。
它們都能把 $x$ 變成一個**比較小**的向量 $z$，再從 $z$ 還原回 $x$。

但如果你問：「**從 $z$ 能不能生成一個從來沒見過的新樣本？**」

兩者的答案都讓人失望。

PCA 的 $z$ 是線性投影的結果，沒有任何機率意義，隨機抽一個 $z$ 丟給重建公式，結果可能合理，也可能完全不像真實資料。
Autoencoder 更嚴重：latent space（潛在空間）的幾何形狀完全由訓練資料決定，裡面沒有任何連續的、有結構的機率分佈，隨手抽的 $z$ 幾乎肯定讓 decoder（解碼器）輸出垃圾。

這就是這篇要處理的核心問題：**怎麼把「壓縮」升級成「推論」，把「重建」升級成「生成」？**

更具體地說，這篇文章想回答四個問題：

1. Standard PCA（標準主成分分析）到底缺了什麼，讓它無法做推論和生成？
2. PPCA（Probabilistic PCA，機率主成分分析）怎麼用一個最簡單的機率生成模型補上這個缺口？
3. PPCA 的三個核心量——邊際分佈、後驗分佈、ML（*Maximum Likelihood*，最大概似）各自在說什麼？
4. 為什麼把線性 decoder 換成神經網路之後，整套推論就崩潰了？這是 VAE（Variational Autoencoder，變分自動編碼器）衍生出來的理由嗎。

如果把整篇濃縮成一句話：

> PPCA 是目前為止最簡單、仍然完整的機率生成模型，它讓你看清楚「可做推論的 latent model」長什麼樣子；
> 而它的限制——linear decoder 才能有 closed-form posterior——剛好定義了下一篇 VAE 要解決的問題。

---

## 為什麼 Standard PCA 不夠用？

### PCA 做什麼

PCA 的目標是找一個低維子空間，讓資料投影進去之後保留最多的變異。具體來說，它找一組正交向量 $w_1, \dots, w_M$，使投影後的方差盡量大：

$$
\text{Var}(w_i^\top x) \text{ 盡量大}
$$

這件事幾何上很直覺：資料的主要變化方向被抓住了，剩餘的方向被當作雜訊丟掉。
投影和重建的公式也很乾淨——硬投影、硬重建，沒有任何隨機性。

### PCA 做不到什麼

正因為 PCA 是純粹的幾何操作，它有三個根本限制：

1. **它沒有噪聲模型:**  PCA 只知道資料「大致落在某個超平面附近」，但對偏離那個平面的部分，它沒有任何機率描述——無法區分「這是合理的觀測噪聲」還是「這是真正不屬於這個空間的樣本」。

2. **它沒辦法做推論（inference）:** 給定一個觀測值 $x$，PCA 可以算出對應的投影 $z = W^\top(x - \bar{x})$，但這個 $z$ 是一個確定的點，不是一個分佈。你沒辦法問「$z$ 的不確定性是多少？」或「$z$ 最可能是哪個範圍？」

3. **它沒有 likelihood（似然）:**  你無法問「這筆資料有多可能是這個模型生成的？」——因為 PCA 根本沒有定義一個機率分佈，自然也就沒有 likelihood。

> **Insight：** 這三個限制都指向同一件事：PCA 是一個幾何最佳化問題，不是一個機率模型。要做推論、要生成新樣本、要比較不同資料的可能性，需要的是一個有完整機率描述的 model，而不只是一條投影公式。

PPCA 就是在 PCA 的骨架上，加入一個最簡單但完整的機率結構，同時修補上面三個問題。

---

## 核心想法：資料是怎麼「被生成」的？

在講 PPCA 的公式之前，先用一個例子建立概念。

你看到的是一張臉的圖片 $x$，解析度是 $128 \times 128$，也就是一個 $D = 16384$ 維的向量。
但這張圖片背後，有幾個你看不見的「解釋因素」在驅動它的內容：

- 這個人的姿勢（頭轉了幾度？）
- 光線的方向（從哪裡照？）
- 是否戴眼鏡？

這些解釋因素的維度 $M$ 遠小於 $D$。它們先存在，再由它們「生成」了你觀測到的高維圖片。

這些看不見的解釋因素，就是 **latent variables**（潛在變數），記作 $z$。

用因果箭頭表示生成方向：

```
z  →  x
```

$z$ 先存在，再根據 $z$ 生成 $x$。這就是 **generative model**（生成模型）的基本骨架，也是 PPCA、VAE、Diffusion 等所有生成模型共用的語言。

> **Note：** 這個 $z \to x$ 的方向是生成方向，訓練時我們做的事情剛好相反——從已知的 $x$ 去推斷背後的 $z$。
> 這個「逆向推斷」就叫做 inference（推論），是潛變數模型最核心的問題。

---

## PPCA 的完整數學推導

### 模型設定

PPCA 把上面的生成故事用 Gaussian（高斯分佈）寫成公式。整個模型由兩層假設構成：

**Latent prior（潛在先驗）：** $z$ 服從標準高斯分佈，維度 $M \ll D$。

$$
p(z) = \mathcal{N}(z;\, 0,\, I)
$$

這個選擇的意思是：在沒有任何觀測資訊之前，我們假設所有 latent factor(variables) 都是相互獨立的、零均值的標準常態變數。

**Observation model（觀測模型）：** 給定 $z$，觀測值 $x$ 由一個線性變換加上 Gaussian 噪聲生成：

$$
p(x \mid z) = \mathcal{N}(x;\, Wz + \mu,\, \sigma^2 I)
$$

這裡的三個元素各有其語意：
- $W \in \mathbb{R}^{D \times M}$：**factor loading matrix（因子載入矩陣）**，把低維 $z$ 線性映射到高維空間，扮演 decoder 的角色
- $\mu \in \mathbb{R}^D$：資料的均值，讓模型不需要先對資料做中心化
- $\sigma^2 I$：各向同性的觀測噪聲，代表「不能被 $M$ 個 latent factor 解釋的部分」

這個噪聲項就是 PCA 一直缺少的東西。
它讓模型能夠區分「$M$ 維子空間以內的訊號」和「之外的噪聲」。

### 邊際分佈：$p(x)$ 長什麼樣子？

模型設定好之後，第一個問題是：如果把 $z$ 積分掉，$x$ 的邊際分佈是什麼？

這個計算直接對 $z$ 做積分：

$$
p(x) = \int p(x \mid z)\, p(z)\, dz
$$

因為 $p(z)$ 和 $p(x \mid z)$ 都是 Gaussian，這個積分有 closed-form（封閉解）。展開計算後：

$$
\boxed{p(x) = \mathcal{N}(x;\, \mu,\, WW^\top + \sigma^2 I)}
$$

這個結果的結構值得細看。協方差矩陣 $WW^\top + \sigma^2 I$ 由兩項組成：

- $WW^\top$：由 latent factors 解釋的變異，是一個 rank-$M$ 的矩陣，只捕捉「主要方向」的相關性
- $\sigma^2 I$：各向同性的殘餘噪聲，對所有維度均等

幾何上，這個分佈的形狀像一個「煎餅」（pancake）：在 $M$ 個主方向上很薄但有厚度 $\sigma^2$，在其他方向上同樣有 $\sigma^2$ 的厚度。
這和 Standard PCA 的「零厚度超平面」形成對比——PPCA 給了資料分佈一個有意義的、有厚度的幾何形狀。

> **Insight：** 這個邊際分佈就是 PPCA 的 likelihood。
> 有了它，你可以問：「給定這組參數 $W, \mu, \sigma^2$，觀測到這批資料 $x$ 的機率是多少？」這是 Standard PCA 完全做不到的事。

### 後驗分佈：給定 $x$，$z$ 的分佈是什麼？

推論問題的方向和生成相反：給定觀測到的 $x$，反推背後的 $z$ 應該長什麼樣子？

這就是計算 **posterior**（後驗分佈）$p(z \mid x)$，用 Bayes rule 可以寫出：

$$
p(z \mid x) = \frac{p(x \mid z)\, p(z)}{p(x)}
$$

因為分子、分母都是 Gaussian，後驗同樣是 Gaussian，可以算出 closed-form：

$$
\boxed{p(z \mid x) = \mathcal{N}\!\left(z;\, M^{-1}W^\top(x - \mu),\, \sigma^2 M^{-1}\right)}
$$

其中 $M = W^\top W + \sigma^2 I \in \mathbb{R}^{M \times M}$。

**這個公式在說什麼？** 給定一個觀測 $x$，對應的 latent code 不是一個點，而是一個分佈。這個分佈的：

- **均值** $M^{-1}W^\top(x - \mu)$：是 $x$ 的線性函數，可以理解為「最可能的 latent code 是什麼」。注意這和 PCA 的硬投影 $W^\top(x - \bar{x})$ 很像，但多了 $M^{-1}$ 這個「收縮因子」，它把估計值往 prior（零點）稍微拉了一下。
- **方差** $\sigma^2 M^{-1}$：代表 latent code 的不確定性。噪聲 $\sigma^2$ 越大，後驗的不確定性越高；觀測 $x$ 能提供的資訊就越少。

> **Insight：** 這個「soft encoding」是 PPCA 相比 PCA 的關鍵優勢。PCA 給你一個點；PPCA 給你一個分佈，包含了不確定性的資訊。這個後驗分佈，就是後面 VAE 中 encoder 要去逼近的目標。

### ML 解：參數要怎麼從資料學出來？

有了模型的定義，下一步是用訓練資料 $\{x_n\}_{n=1}^N$ 估計參數 $W, \mu, \sigma^2$。

對 $\mu$ 的 ML 估計很直觀：就是樣本均值 $\hat{\mu} = \frac{1}{N}\sum_n x_n$。

對 $W$ 和 $\sigma^2$ 的 ML 估計，可以透過最大化 log-likelihood $\sum_n \log p(x_n)$ 得到。
令 $S = \frac{1}{N}\sum_n (x_n - \mu)(x_n - \mu)^\top$ 為樣本協方差矩陣，$\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_D$ 為其特徵值，對應特徵向量為 $u_1, \dots, u_D$，則：

$$
W_{ML} = U_M \left(\Lambda_M - \sigma^2 I\right)^{1/2} R
$$

$$
\boxed{\sigma^2_{ML} = \frac{1}{D - M} \sum_{i=M+1}^{D} \lambda_i}
$$

其中 $U_M = [u_1, \dots, u_M]$ 是前 $M$ 個特徵向量排成的矩陣，$\Lambda_M = \mathrm{diag}(\lambda_1, \dots, \lambda_M)$，$R$ 是任意正交矩陣。

**$\sigma^2_{ML}$ 的意義是什麼？** 它等於被丟掉的那 $D - M$ 個方向上的特徵值平均。直觀地說：把前 $M$ 個主方向用 $W$ 捕捉之後，剩下的方向的平均變異量，就是模型估計的「各向同性噪聲水準」。

這個結果有一個漂亮的邊界行為：當 $M \to D$ 時，所有方向都被 $W$ 捕捉，$\sigma^2_{ML} \to 0$，模型退化回沒有噪聲的情況，也就是 Standard PCA。

**$R$ 帶來旋轉不唯一性** ——PPCA 能識別的是 $W$ 張成的子空間，不是 $W$ 的具體列向量方向。因為對任意正交矩陣 $R$，替換 $W \leftarrow WR$ 不影響模型輸出：

$$
(WR)(WR)^\top = WRR^\top W^\top = WW^\top
$$

$p(x)$ 完全不變。這和 Standard PCA 中「特徵向量可以翻轉方向」是同一個現象——識別的是子空間，不是具體的 basis。

---

## EM 框架：把推論和學習分開看

### 為什麼要學 EM？

PPCA 的參數有 closed-form ML 解，理論上不需要迭代。但學 **EM（Expectation-Maximization，期望最大化）** 演算法在 PPCA 上的形式，有一個更重要的原因：EM 把「潛變數模型怎麼學」這件事拆成兩個步驟，這個思路在後面更複雜的模型（包括 VAE）裡會反覆出現，只是形式有所不同。

### 問題的根源

訓練的目標是最大化觀測資料的 log-likelihood：

$$
\log p(X;\theta) = \log \int p(X, Z;\theta)\, dZ
$$

問題在這個積分——當 decoder 是 nonlinear 神經網路時，integrand 不再是 Gaussian，這個積分沒有 closed-form，直接最大化也算不了梯度。EM 的作用是繞開這個積分，把難題改寫成可以交替優化的形式。

### 下界的推導

引入任意輔助分佈 $q(Z)$，對 $\log$ 用 **Jensen's inequality（詹森不等式）** 做下界：

$$
\log p(X;\theta) = \log \int q(Z)\frac{p(X,Z;\theta)}{q(Z)}\, dZ
\ge \int q(Z)\log \frac{p(X,Z;\theta)}{q(Z)}\, dZ =: \mathcal{L}(q, \theta)
$$

下界和真實 log-likelihood 的差距，恰好等於一個 **KL divergence（KL 散度）**：

$$
\log p(X;\theta) = \mathcal{L}(q, \theta) + \underbrace{\mathrm{KL}\!\left(q(Z) \,\|\, p(Z \mid X;\theta)\right)}_{\ge 0}
$$

當且僅當 $q(Z) = p(Z \mid X;\theta)$ 時，KL 項等於零，下界緊貼真實 log-likelihood。

### 兩個交替步驟

EM 利用上面的結構，設計了一個交替優化的流程：

**E-step（Expectation step，期望步驟）：** 固定參數 $\theta$，令 $q(Z) = p(Z \mid X;\theta_{old})$。此時 KL 項歸零，下界完全等於 log-likelihood。這一步的本質是「用目前的模型做一次推論，算出 latent variables 的後驗分佈」。

**M-step（Maximization step，最大化步驟）：** 固定 $q$，最大化 $\mathcal{L}(q, \theta)$ 更新參數 $\theta$。因為 $q$ 被固定住了，直接對 $\theta$ 求最大值，得到新的參數估計。

> **Note：** 這個「E-step 做 inference，M-step 做 parameter update」的骨架，在 PPCA 裡兩個步驟都有解析解。
> 但在 VAE 裡，E-step 的精確 posterior 算不出來，所以要用 encoder 網路近似；M-step 也沒有解析解，改用梯度下降。
> 這就是 PPCA 和 VAE 最根本的訓練差異。

### PPCA 的 EM 更新式

**E-step**——利用前面推導的後驗分佈，計算 latent variables 的充分統計量：

$$
\mathbb{E}[z_n] = M_{old}^{-1} W_{old}^\top (x_n - \mu)
$$

$$
\mathbb{E}[z_n z_n^\top] = \sigma^2_{old}\, M_{old}^{-1} + \mathbb{E}[z_n]\mathbb{E}[z_n]^\top
$$

其中 $M_{old} = W_{old}^\top W_{old} + \sigma^2_{old} I$。

**M-step**——用 E-step 算出的統計量更新 $W$ 和 $\sigma^2$：

$$
W_{new} = \left[\sum_{n=1}^N (x_n - \mu)\,\mathbb{E}[z_n]^\top\right] \left[\sum_{n=1}^N \mathbb{E}[z_n z_n^\top]\right]^{-1}
$$

$$
\sigma^2_{new} = \frac{1}{ND}\sum_{n=1}^N \Bigl\{ \|x_n - \mu\|^2 - 2\,\mathbb{E}[z_n]^\top W_{new}^\top (x_n - \mu) + \mathrm{Tr}\!\left(\mathbb{E}[z_n z_n^\top]\, W_{new}^\top W_{new}\right) \Bigr\}
$$

這兩個更新式對初學者可以不需要記住細節，但要理解結構：M-step 的 $W_{new}$ 的形式是「觀測與 latent 的協方差，除以 latent 的二階矩」——本質上就是一個 regression 問題的解，只是「標籤」是 latent variable 的後驗期望值。

---

## Standard PCA 與 PPCA 的本質差別

把兩者放在一起比較，差異一目瞭然：

| 面向 | Standard PCA | Probabilistic PCA（PPCA） |
|---|---|---|
| 性質 | Deterministic 幾何投影 | Stochastic 生成模型 |
| Encoding（編碼） | $z = W^\top(x - \bar{x})$，硬投影，輸出一個點 | $p(z \mid x) = \mathcal{N}(M^{-1}W^\top(x-\mu),\, \sigma^2 M^{-1})$，輸出一個分佈 |
| Decoding（解碼） | $\hat{x} = Wz + \bar{x}$，硬重建 | $p(x \mid z) = \mathcal{N}(Wz + \mu,\, \sigma^2 I)$，隨機生成 |
| 噪聲模型 | 無 | 有（$\sigma^2 I$），對偏離子空間的部分有明確描述 |
| Likelihood | 無法定義 | $p(x) = \mathcal{N}(\mu,\, WW^\top + \sigma^2 I)$ |
| 生成新樣本 | 不自然（沒有定義 $p(z)$） | 先抽 $z \sim p(z)$，再從 $p(x \mid z)$ 採樣 |
| 幾何形狀 | 零厚度超平面 | 有厚度的 pancake-shaped 高機率流形 |
| $\sigma^2 \to 0$ 的極限 | — | 退化回 Standard PCA |

最後一行是整個 PPCA 框架最重要的性質：**Standard PCA 是 PPCA 在 $\sigma^2 \to 0$ 時的特例。**
PPCA 不是「換個說法的 PCA」，而是真正包含了 PCA，同時提供更豐富的機率語言。

---

## 從 PPCA 到 VAE：一個改動，一道鴻溝

這一節是這篇文章最重要的橋段——它解釋了為什麼 PPCA 是「VAE 的前身」，以及這個前身在哪裡碰了壁。

### 只改了一件事

比較 PPCA 和 VAE 的 decoder：

**PPCA 的 decoder（線性）：**

$$
p(x \mid z) = \mathcal{N}(x;\; Wz + \mu,\; \sigma^2 I)
$$

**VAE 的 decoder（非線性）：**

$$
p_\theta(x \mid z) = \mathcal{N}(x;\; f_\theta(z),\; \sigma^2 I)
$$

唯一的差別是把線性映射 $Wz + \mu$ 換成了神經網路 $f_\theta(z)$。這個改動讓 decoder 的表達能力從線性跳到任意非線性，理論上能夠建模複雜得多的資料分佈。

但代價是什麼？

### Posterior 從此算不出來

PPCA 的後驗之所以有 closed-form，是因為 $p(z)$ 和 $p(x \mid z)$ 都是 $z$ 的 **quadratic form（二次型）**——指數部分是 $z$ 的二次式，兩者相乘仍然是二次式，所以後驗還是 Gaussian。

可以把 PPCA 的 posterior log-density 展開確認這件事：

$$
\log p(z \mid x)
= -\frac{1}{2} z^\top z - \frac{1}{2\sigma^2} \|x - Wz - \mu\|^2 + \text{const}
$$

展開平方項，所有 $z$ 的項都只到二次：

$$
= -\frac{1}{2} z^\top \!\left(I + \frac{W^\top W}{\sigma^2}\right)\! z + \frac{1}{\sigma^2} z^\top W^\top(x - \mu) + \text{const}
$$

這就是 Gaussian 的 log-density 形式，後驗確實是 Gaussian，有精確的解析解。

一旦把 $Wz + \mu$ 換成 nonlinear $f_\theta(z)$，指數部分變成：

$$
-\frac{1}{2\sigma^2}\|x - f_\theta(z)\|^2
$$

這裡藏了 $f_\theta(z)$，是 $z$ 的非線性函數，整個 exponent 不再是 $z$ 的二次型，後驗分佈的形狀可以任意複雜，不屬於任何有 closed-form 的分佈族。

> **Insight：** 這就是 VAE 需要 encoder 網路 $q_\phi(z \mid x)$ 的根本原因。VAE 的 encoder 不是在做「壓縮」，而是在用一個參數化的 Gaussian 分佈，去近似那個算不出來的真實 posterior。
>
> PPCA：posterior 精確可算，encoder = closed-form 公式
> VAE：posterior 不可解析，encoder = 用神經網路學出來的近似分佈

用一張對比圖把兩者的推論方式說清楚：

```
PPCA（線性 decoder）
  ┌──────────────────────────────────────────────────┐
  │  p(z|x) = N( M⁻¹Wᵀ(x-μ), σ²M⁻¹ )               │
  │  精確算出來，不需要任何額外模型                   │
  │  給定 x，z 的分佈的 mean 和 variance 都有公式     │
  └──────────────────────────────────────────────────┘

VAE（非線性 decoder，f_θ 是神經網路）
  ┌──────────────────────────────────────────────────┐
  │  p_θ(z|x) 算不出來                               │
  │  改用 q_φ(z|x) = N( μ_φ(x), diag(σ²_φ(x)) ) 近似│
  │  μ_φ 和 σ_φ 都是 encoder 神經網路的輸出           │
  └──────────────────────────────────────────────────┘
```

這個從「精確推論」到「近似推論」的跨越，是整個深層生成模型的分水嶺。
PPCA 活在精確推論可行的線性世界；VAE 把自己推進非線性的世界，換來的是更強的生成能力，付出的代價是需要學習一個 approximate inference 機制。

---

## 常見誤解

**誤解 1：PPCA 和 PCA 只是換個說法**

完全不同。PCA 是幾何投影，沒有噪聲模型、沒有 likelihood、沒有後驗分佈。PPCA 是一個完整的機率生成模型，能回答「這筆資料有多可能是這個模型生成的？」，能計算 latent code 的不確定性，也能從 prior 採樣生成新的觀測值。PCA 都做不到。

**誤解 2：Latent variable 一定對應可解釋的語意**

不一定。PPCA 只保證 latent factors 能解釋資料的 covariance 結構，不保證每個維度自動對應「眼鏡」「角度」這種人類語言。Disentanglement（解糾纏）是需要額外設計的問題，不是 latent variable model 的內建性質。

**誤解 3：旋轉不唯一性代表模型不穩定**

旋轉不唯一是設計上的固有性質，不是問題。它只是說 PPCA 識別的是 $W$ 張成的子空間，而不是 $W$ 的具體列向量方向。只要下游任務依賴的是整個 subspace（例如重建），旋轉不影響任何結果。如果你需要可解釋的 latent axes，要加額外約束（例如 ICA，Independent Component Analysis，獨立成分分析）。

**誤解 4：EM 只是 PPCA 的一種訓練方式**

EM 是所有潛變數模型的通用學習框架，PPCA 只是它剛好有 closed-form 解的特殊案例。學 EM 真正要記住的不是更新式本身，而是那個思路：E-step 做推論（inference），M-step 更新參數，交替直到收斂。這個骨架在 VAE 裡還會出現，只是 E-step 從精確的後驗計算，變成 encoder 網路的 forward pass。

---

## 總結

1. **Standard PCA 的三個缺口**：沒有噪聲模型、沒辦法做推論（posterior）、沒有 likelihood。這三件事讓 PCA 無法做生成，也無法量化不確定性。

2. **PPCA 的生成故事**：$z \sim \mathcal{N}(0, I)$ 先存在，再由 $x \mid z \sim \mathcal{N}(Wz + \mu, \sigma^2 I)$ 生成觀測值。這個結構讓 PPCA 同時擁有可算的 likelihood、可做推論的 posterior，以及可生成新樣本的生成流程。

3. **三個核心量**：
   - 邊際分佈 $p(x) = \mathcal{N}(\mu,\, WW^\top + \sigma^2 I)$：訊號與噪聲的疊加，是 PPCA 的 likelihood。
   - 後驗分佈 $p(z \mid x) = \mathcal{N}(M^{-1}W^\top(x-\mu),\, \sigma^2 M^{-1})$：給定觀測，latent code 的不確定性分佈，是「soft encoding」。
   - ML 解 $\sigma^2_{ML} = \frac{1}{D-M}\sum_{i=M+1}^D \lambda_i$：被丟掉的方向的平均變異量，就是噪聲的 ML 估計。

4. **EM 的骨架**：E-step 做 inference（算 posterior），M-step 做 parameter update（最大化 lower bound）。這個「推論與學習交替」的框架，是理解 VAE 訓練邏輯的直接前身。

5. **PPCA 到 VAE 的分水嶺**：把線性 decoder 換成神經網路，後驗從 Gaussian closed-form 變成不可解析的任意形狀，迫使 VAE 必須引入 encoder 做 approximate inference。這個「一個改動帶來的後果」，定義了下一篇要解決的核心問題。
