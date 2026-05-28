---
type: Blog-post
tags:
  - Deep-Learning
  - note
created: 2026-05-27
updated: 2026-05-28
authors: WeiTa
status: active
---

# Linear Factor Models 與 Probabilistic PCA (PPCA)

## 這篇文章想回答什麼問題？

PCA 大多數人都學過，但通常是從線性代數的角度理解：找一組正交基底，讓投影後的資料變異數最大。

這篇文章想把同一件事改寫成機率語言：如果資料背後真的有一組看不見的 latent factors 在驅動，這個生成故事應該怎麼寫？只觀察到 $x$、看不到 $z$ 的時候，模型還能不能被學出來？

> 如果資料背後真的有一組我們看不見的 latent factors 在驅動，那這件事要怎麼用機率語言說清楚？只觀察到 $x$ 卻看不到 $z$，模型還能不能被學出來？

更具體地說，這篇文章想回答三個問題：

1. **Latent variable model 的生成故事是什麼？** 資料 $x$ 是怎麼從 $z$ 產生的？
2. **PPCA 和 Standard PCA 的本質差別在哪？** 為什麼「加上機率」不只是換個說法？
3. **為什麼 PPCA 是通往 VAE 的橋樑？** linear decoder 換成 nonlinear 之後，什麼東西開始壞掉？

如果把這篇濃縮成一句話：

> PPCA 用一個最簡單、但仍然完整的機率生成模型，把 PCA 從幾何方法提升成可以做推論、做學習、做生成的 probabilistic model；而它真正的價值，是讓你看清楚 VAE 到底在改什麼。

在系列位置上，這篇是從「線性降維」走向「潛變數生成模型」的第一個完整橋段。你可以把它看成後面 autoencoder、VAE、甚至更一般 latent variable model 的底層語法。

## 摘要

- PPCA 是在 PCA 之上加入機率化（噪聲模型與 likelihood），因此能做推論與生成。
- 在線性 decoder 的情況下，PPCA 的 posterior 與 ML 解都有封閉解；把 decoder 換成非線性（例如神經網路）後，就引出 VAE 的必要性。
- EM 的思路（E-step 做 inference，M-step 更新參數）是理解更複雜潛變數模型的重要橋樑。


---

## 為什麼 Standard PCA 不夠用？

### PCA 的目標是什麼？

PCA 的核心是找一個低維子空間，讓資料投影進去之後保留最多的變異數。具體來說，它會找一組正交向量 $w_1, \dots, w_M$，讓投影後的方差盡量大：

$$
\text{Var}\left(w_i^T x\right) \text{ 盡量大}
$$

這件事的幾何圖像很乾淨：資料的主要變化方向被抓到了，次要的 noise 被丟掉了。若只看重建，PCA 的結果也很直覺：把每筆資料投到前 $M$ 個主成分張成的子空間，再投影回原空間。

### PCA 做不到什麼？

但 PCA 有幾件事做不到：

**第一，它沒有噪聲模型。** PCA 只知道資料「大約落在某個超平面上」，但說不清楚偏離那個平面的部分是噪聲還是訊號。沒有噪聲模型，就沒辦法做推論——你無法說「給定觀測 $x$，latent code $z$ 的分佈是什麼」。

**第二，它不能自然地生成新樣本。** 你可以把 $x$ 投影到 $z$，但如果我給你一個 $z$，你怎麼生成一個「合理的」$x$？PCA 是 deterministic 的，這個問題沒有清楚答案。

**第三，它沒有 likelihood。** 你沒辦法問「這筆資料符合這個模型的機率是多少」，因為根本沒有定義機率。也就是說，PCA 是一個幾何最佳化問題，但不是一個完整的 probabilistic model。

PPCA 就是要把上面三件事都修補起來。

---

## 核心想法：用機率語言說生成故事

### 資料是怎麼來的？

先看最直觀的影像例子。你看到的是一張臉的圖片 $x$，但圖片背後有一些你看不見的解釋因素：

- 這個人有沒有戴眼鏡？
- 光線從哪個方向來？
- 頭轉了幾度？

這些「解釋因素」就是 latent variables $z$。它們驅動了觀測資料的生成。

用一個箭頭表示因果關係：

```
z  →  x
```

$z$ 先存在，再根據 $z$ 生成 $x$。這就是 generative model 的基本骨架。

### PPCA 的具體設定

PPCA 把這個故事用 Gaussian 寫成公式。它最基本的假設可以分成兩層：先定義 latent prior，再定義觀測的條件分佈。

**Latent prior：**

$$
\boxed{\sigma^2_{ML} = \frac{1}{D - M}\sum_{i=M+1}^{D} \lambda_i}
$$

其中：
- $U_M$：前 $M$ 個 principal eigenvectors
- $\Lambda_M = \mathrm{diag}(\lambda_1, \dots, \lambda_M)$
- $R$：任意 orthogonal matrix（旋轉不唯一性，後面會說）

**$\sigma^2_{ML}$ 的幾何意義：** 它是「沒有被前 $M$ 個主成分解釋掉的特徵值的平均」，也就是剩餘 noise 的平均水準。若 $M = D-1$，則 $\sigma^2_{ML} = \lambda_D$；若 $M \to D$，$\sigma^2 \to 0$。

### 為什麼會有旋轉不唯一性？

要注意的是，PPCA 真正識別的是 subspace，不是 basis 的絕對方向。因為若 $R$ 是任意 orthogonal matrix，則

$$
(WR)(WR)^T = W R R^T W^T = WW^T
$$

所以 $p(x)$ 完全不受影響。這和 standard PCA 一樣：eigenvectors 本身可以旋轉，但它們張成的子空間不變。

---

## 當 closed-form 解不好直接求時：EM 的觀點

PPCA 的 ML 解可以寫成 closed-form，但 EM 還是值得單獨看，因為它把「潛變數模型怎麼學」這件事拆成兩個可理解的步驟。這個拆法，會直接變成後面 VAE 的核心骨架。

### 為什麼需要 EM？

直接最大化 $\log p(X;\theta)$ 要計算：

$$
p(X;\theta) = \int p(X, Z;\theta)\, dZ
$$

當 decoder 是 nonlinear（例如神經網路）時，這個積分通常沒有 closed-form。EM 的作用，就是把原本難算的 marginal likelihood，改寫成一個可交替優化的 lower bound。

### 下界的推導

引入任意輔助分佈 $q(Z)$，用 Jensen inequality 對 log 做下界：

$$
\log p(X;\theta) = \log \int q(Z)\frac{p(X,Z;\theta)}{q(Z)}\, dZ
\ge \int q(Z)\log \frac{p(X,Z;\theta)}{q(Z)}\, dZ =: \mathcal{L}(q, \theta)
$$

更精確地，等號成立當 $q(Z) = p(Z \mid X;\theta)$，此時：

$$
\log p(X;\theta) = \mathcal{L}(q, \theta) + \underbrace{\mathrm{KL}(q(Z) \,\|\, p(Z \mid X;\theta))}_{\ge 0}
$$

因此 EM 會交替做兩件事：

**E-step：** 固定 $\theta$，令 $q(Z) = p(Z \mid X;\theta_{old})$，讓下界緊貼真實 log-likelihood。

**M-step：** 固定 $q$，最大化 $\mathcal{L}(q, \theta)$ 更新參數 $\theta$。

### PPCA 的 EM 更新式

對單筆資料 $x_n$，E-step 用前面算出的 posterior 直接得到充分統計量：

$$
\mathbb{E}[z_n] = M_{old}^{-1} W_{old}^T (x_n - \mu)
$$

$$
\mathbb{E}[z_n z_n^T] = \sigma^2_{old}\, M_{old}^{-1} + \mathbb{E}[z_n]\mathbb{E}[z_n]^T
$$

其中 $M_{old} = W_{old}^T W_{old} + \sigma^2_{old} I$。

在 M-step 裡，把這些期望代回 complete-data log-likelihood，對 $W$ 與 $\sigma^2$ 做最大化，會得到：

$$
W_{new} = \left[\sum_{n=1}^N (x_n - \mu)\,\mathbb{E}[z_n]^T\right] \left[\sum_{n=1}^N \mathbb{E}[z_n z_n^T]\right]^{-1}
$$

$$
\sigma^2_{new} = \frac{1}{ND}\sum_{n=1}^N \left\{ \|x_n - \mu\|^2 - 2\,\mathbb{E}[z_n]^T W_{new}^T (x_n - \mu) + \mathrm{Tr}\!\left(\mathbb{E}[z_n z_n^T]\, W_{new}^T W_{new}\right) \right\}
$$

> **Note：** PPCA 有 closed-form 解，所以不需要真的靠 EM 才能訓練；但 EM 的觀點非常重要，因為它就是「推論先行、再更新參數」的基本模板。後面的 VAE 本質上是在做同一件事，只是 E-step 變成 approximate inference。

---

## Standard PCA 與 PPCA 的本質差別

| 面向 | Standard PCA | Probabilistic PCA |
|-----|-------------|------------------|
| 性質 | Deterministic 幾何投影 | Stochastic 生成模型 |
| Encoder | $z = W^T(x - \bar{x})$（硬投影） | $p(z \mid x) = \mathcal{N}(M^{-1}W^T(x-\mu),\, \sigma^2 M^{-1})$ |
| Decoder | $\hat{x} = Wz + \bar{x}$（硬重建） | $p(x \mid z) = \mathcal{N}(Wz + \mu,\, \sigma^2 I)$ |
| 噪聲模型 | 無 | 有（$\sigma^2 I$） |
| Likelihood | 無法定義 | $p(x) = \mathcal{N}(\mu,\, WW^T + \sigma^2 I)$ |
| 生成新樣本 | 不自然 | 先抽 $z \sim p(z)$，再算 $p(x \mid z)$ |
| 幾何解讀 | 零厚度超平面 | 有厚度的 pancake-shaped 高機率流形 |
| $\sigma^2 \to 0$ 極限 | — | 退化回 Standard PCA |

最後一行是關鍵：**Standard PCA 是 PPCA 在 $\sigma^2 \to 0$ 時的特例。**

---

## PPCA 到 VAE：一步之遙，一鴻溝之差

### 為什麼說 PPCA 是 VAE 的前身？

看它們的生成結構：

**PPCA：**

$$
p(x \mid z) = \mathcal{N}(x;\; Wz + \mu,\; \sigma^2 I)
$$

**VAE：**

$$
p_\theta(x \mid z) = \mathcal{N}(x;\; f_\theta(z),\; \sigma^2 I)
$$

唯一的差別是把線性 decoder $Wz + \mu$ 換成了神經網路 $f_\theta(z)$。但這個改動的後果很大：原本可解析的 posterior，會立刻變成算不出來的分佈。

### Posterior 為什麼在 VAE 裡算不出來？

PPCA 的 posterior 是 Gaussian，是因為 $p(z)$ 和 $p(x \mid z)$ 的 exponent 都是 $z$ 的 quadratic form，相乘後仍然是 quadratic，因此 posterior 仍然是 Gaussian。

一旦 decoder 換成 nonlinear network $f_\theta(z)$，$p(x \mid z)$ 的 exponent 變成：

$$
-\frac{1}{2\sigma^2}\|x - f_\theta(z)\|^2
$$

這裡面藏了 $f_\theta(z)$，是 $z$ 的 nonlinear 函數，整個 exponent 不再是 $z$ 的 quadratic form，因此 posterior 不再是 Gaussian，也沒有 closed-form。

> **Insight：** 這就是為什麼 VAE 需要「encoder 網路 $q_\phi(z \mid x)$」—— 它不是真正的 posterior，而是用來近似那個算不出來的 posterior 的東西。PPCA 有精確 posterior，VAE 只有 approximate posterior。代價換來的，是更強的生成能力。

### 兩者的推論方式對比

```
PPCA
  ┌──────────────────────────────────────────┐
  │  p(z|x) = N(M⁻¹Wᵀ(x-μ), σ²M⁻¹)         │
  │  精確算出來，不需要額外模型               │
  └──────────────────────────────────────────┘

VAE
  ┌──────────────────────────────────────────┐
  │  p(z|x) 算不出來                         │
  │  改用 q_φ(z|x) = N(μ_φ(x), σ²_φ(x)) 近似│
  │  μ_φ, σ_φ 都是 neural network 的輸出     │
  └──────────────────────────────────────────┘
```

---

## 常見誤解

**誤解 1：Latent variable 一定對應可解釋的語意**

不一定。模型只保證 latent factors 能解釋資料的 covariance 結構，不保證每個維度自動對應到「眼鏡」「性別」「姿勢」這種人類語言。Disentanglement 是另一個要額外設計的問題。

**誤解 2：PPCA 和 PCA 只是換個說法**

完全不一樣。PCA 只能做 projection，PPCA 有 likelihood、有 posterior、有噪聲模型，能做推論和生成。換個說法，PPCA 能回答「這筆資料有多可能是這個模型生成的？」，PCA 回答不了。

**誤解 3：旋轉不唯一性代表模型學不穩**

旋轉不唯一是設計上的固有性質，不是 bug。它說明模型識別的是 subspace，不是 basis 的方向。只要 downstream task 依賴的是 subspace 本身（例如 reconstruction），旋轉不影響結果。但如果你想要可解釋的 latent axes，就需要加額外約束（例如 independent component analysis 那條路）。

**誤解 4：EM 只是 PPCA 的一種訓練方式**

更準確地說，EM 是 latent variable model 的通用學習框架，PPCA 只是它有 closed-form 解的特殊案例。當你學 PPCA 的 EM，真正要記住的是那個思路：E-step 做 inference，M-step 做 parameter update，交替直到收斂。這個骨架在後面的 VAE 裡還會出現，只是 E-step 變成了 encoder 的 forward pass。

---

## 總結

1. **PPCA 解決的根本問題**：讓 PCA 從幾何投影升級成有完整機率描述的生成模型，同時保留 tractable inference 和 closed-form 解。

2. **三個核心推導**：
   - 邊際分佈 $p(x) = \mathcal{N}(\mu,\, WW^T + \sigma^2 I)$ —— signal 和 noise 的疊加。
   - 後驗分佈 $p(z \mid x) = \mathcal{N}(M^{-1}W^T(x-\mu),\, \sigma^2 M^{-1})$ —— 精確的 soft encoding。
   - ML 解 $\sigma^2_{ML} = \frac{1}{D-M}\sum_{i=M+1}^D \lambda_i$ —— 被丟掉的特徵值的平均就是噪聲。

3. **Standard PCA 是 $\sigma^2 \to 0$ 的特例**：PPCA 多了厚度（噪聲），多了機率，但少了那份幾何簡潔。

4. **PPCA 到 VAE 的關鍵跨越**：把線性 decoder 換成 nonlinear network，後驗從 closed-form 變成 intractable，這迫使 VAE 必須引入 encoder 網路做 approximate inference。Closed-form 的代價，正是 PPCA 表達能力的天花板。

5. **EM 的角色**：PPCA 有 closed-form 所以不需要跑 EM，但 EM 的思路（E-step inference，M-step update）就是 VAE training loop 的原型，只是把精確的步驟全部換成近似版。

這篇文章的收束很簡單：PPCA 不是 PCA 的修飾版，而是一個完整的 linear latent variable model。下一篇如果把 decoder 從 linear 改成 nonlinear，就會自然走到 autoencoder 與 VAE，因為真正開始困難的地方，就是 posterior 不再能解析地算出來。

---

## 當 Closed-Form 解不存在：EM 算法

PPCA 有 closed-form 解是因為 decoder 是線性的。但引入 EM 的思路很重要，因為它會直接連到後來的 VAE。

### 為什麼需要 EM？

直接最大化 $\log p(X;\theta)$ 要計算：

$$
p(X;\theta) = \int p(X, Z;\theta)\, dZ
$$

當 decoder 是 nonlinear（例如神經網路）時，這個積分沒有 closed-form。EM 把這個問題繞開了。

### 下界的推導

引入任意輔助分佈 $q(Z)$，用 Jensen inequality 對 log 做下界：

$$
\log p(X;\theta) \ge \int q(Z) \log \frac{p(X, Z;\theta)}{q(Z)}\, dZ =: \mathcal{L}(q, \theta)
$$

更精確地，等號成立當 $q(Z) = p(Z \mid X;\theta)$，此時：

$$
\log p(X;\theta) = \mathcal{L}(q, \theta) + \underbrace{\mathrm{KL}(q(Z) \,\|\, p(Z \mid X;\theta))}_{\ge 0}
$$

EM 交替做兩件事：

**E-step：** 固定 $\theta$，令 $q(Z) = p(Z \mid X;\theta_{old})$，讓下界緊貼真實 log-likelihood。

**M-step：** 固定 $q$，最大化 $\mathcal{L}(q, \theta)$ 更新參數 $\theta$。

### PPCA 的 EM 更新式

**E-step** 計算後驗的充分統計量（利用前面的 $p(z \mid x)$ 結果）：

$$
\mathbb{E}[z_n] = M_{old}^{-1} W_{old}^T (x_n - \mu)
$$

$$
\mathbb{E}[z_n z_n^T] = \sigma^2_{old}\, M_{old}^{-1} + \mathbb{E}[z_n]\mathbb{E}[z_n]^T
$$

其中 $M_{old} = W_{old}^T W_{old} + \sigma^2_{old} I$。

**M-step** 用 E-step 的結果更新參數：

$$
W_{new} = \left[\sum_{n=1}^N (x_n - \mu)\,\mathbb{E}[z_n]^T\right] \left[\sum_{n=1}^N \mathbb{E}[z_n z_n^T]\right]^{-1}
$$

$$
\sigma^2_{new} = \frac{1}{ND}\sum_{n=1}^N \left\{ \|x_n - \mu\|^2 - 2\,\mathbb{E}[z_n]^T W_{new}^T (x_n - \mu) + \mathrm{Tr}\!\left(\mathbb{E}[z_n z_n^T]\, W_{new}^T W_{new}\right) \right\}
$$

> **Note：** PPCA 有 closed-form 解，所以不需要跑 EM；但 EM 的思路對後面的 VAE 非常關鍵。EM 是「精確 posterior + 精確 maximization」，VAE 則是「approximate posterior（encoder）+ gradient-based maximization」。

---

## Standard PCA 與 PPCA 的本質差別

| 面向 | Standard PCA | Probabilistic PCA |
|-----|-------------|------------------|
| 性質 | Deterministic 幾何投影 | Stochastic 生成模型 |
| Encoder | $z = W^T(x - \bar{x})$（硬投影） | $p(z \mid x) = \mathcal{N}(M^{-1}W^T(x-\mu),\, \sigma^2 M^{-1})$ |
| Decoder | $\hat{x} = Wz + \bar{x}$（硬重建） | $p(x \mid z) = \mathcal{N}(Wz + \mu,\, \sigma^2 I)$ |
| 噪聲模型 | 無 | 有（$\sigma^2 I$） |
| Likelihood | 無法定義 | $p(x) = \mathcal{N}(\mu,\, WW^T + \sigma^2 I)$ |
| 生成新樣本 | 不自然 | 先抽 $z \sim p(z)$，再算 $p(x \mid z)$ |
| 幾何解讀 | 零厚度超平面 | 有厚度的 pancake-shaped 高機率流形 |
| $\sigma^2 \to 0$ 極限 | — | 退化回 Standard PCA |

最後一行是關鍵：**Standard PCA 是 PPCA 在 $\sigma^2 \to 0$ 時的特例。**

---

## PPCA 到 VAE：一步之遙，一鴻溝之差

### 為什麼說 PPCA 是 VAE 的前身？

看它們的生成結構：

**PPCA：**

$$
p(x \mid z) = \mathcal{N}(x;\; Wz + \mu,\; \sigma^2 I)
$$

**VAE：**

$$
p_\theta(x \mid z) = \mathcal{N}(x;\; f_\theta(z),\; \sigma^2 I)
$$

唯一的差別是把線性 decoder $Wz + \mu$ 換成了神經網路 $f_\theta(z)$。

但這「一個改動」帶來了截然不同的後果：

### Posterior 為什麼在 VAE 裡算不出來？

PPCA 的 posterior 是 Gaussian 是因為 $p(z)$ 和 $p(x \mid z)$ 的 exponent 都是 $z$ 的 quadratic form，相乘後仍然是 quadratic，因此 posterior 仍然是 Gaussian。

一旦 decoder 換成 nonlinear network $f_\theta(z)$，$p(x \mid z)$ 的 exponent 變成：

$$
-\frac{1}{2\sigma^2}\|x - f_\theta(z)\|^2
$$

這裡面藏了 $f_\theta(z)$，是 $z$ 的 nonlinear 函數，整個 exponent 不再是 $z$ 的 quadratic form，posterior 不再是 Gaussian，也沒有 closed-form。

> **Insight：** 這就是為什麼 VAE 需要「encoder 網路 $q_\phi(z \mid x)$」—— 它不是真正的 posterior，而是用來近似那個算不出來的 posterior 的東西。PPCA 有精確 posterior，VAE 只有 approximate posterior。代價換來的，是更強的生成能力。


### 兩者的推論方式對比

```
PPCA
  ┌──────────────────────────────────────────┐
  │  p(z|x) = N(M⁻¹Wᵀ(x-μ), σ²M⁻¹)         │
  │  精確算出來，不需要額外模型               │
  └──────────────────────────────────────────┘

VAE
  ┌──────────────────────────────────────────┐
  │  p(z|x) 算不出來                         │
  │  改用 q_φ(z|x) = N(μ_φ(x), σ²_φ(x)) 近似│
  │  μ_φ, σ_φ 都是 neural network 的輸出     │
  └──────────────────────────────────────────┘
```

---

## 常見誤解

**誤解 1：Latent variable 一定對應可解釋的語意**

不一定。模型只保證 latent factors 能解釋資料的 covariance 結構，不保證每個維度自動對應到「眼鏡」「性別」「姿勢」這種人類語言。Disentanglement 是另一個要額外設計的問題。

**誤解 2：PPCA 和 PCA 只是換個說法**

完全不一樣。PCA 只能做 projection，PPCA 有 likelihood、有 posterior、有噪聲模型，能做推論和生成。換個說法，PPCA 能回答「這筆資料有多可能是這個模型生成的？」，PCA 回答不了。

**誤解 3：旋轉不唯一性代表模型學不穩**

旋轉不唯一是設計上的固有性質，不是 bug。它說明模型識別的是 subspace，不是 basis 的方向。只要 downstream task 依賴的是 subspace 本身（例如 reconstruction），旋轉不影響結果。但如果你想要可解釋的 latent axes，就需要加額外約束（例如 independent component analysis 那條路）。

**誤解 4：EM 只是 PPCA 的一種訓練方式**

更準確地說，EM 是 latent variable model 的通用學習框架，PPCA 只是它有 closed-form 解的特殊案例。當你學 PPCA 的 EM，真正要記住的是那個思路：E-step 做 inference，M-step 做 parameter update，交替直到收斂。這個骨架在後面的 VAE 裡還會出現，只是 E-step 變成了 encoder 的 forward pass。

---

## 總結

1. **PPCA 解決的根本問題**：讓 PCA 從幾何投影升級成有完整機率描述的生成模型，同時保留 tractable inference 和 closed-form 解。

2. **三個核心推導**：
   - 邊際分佈 $p(x) = \mathcal{N}(\mu,\, WW^T + \sigma^2 I)$ —— signal 和 noise 的疊加。
   - 後驗分佈 $p(z \mid x) = \mathcal{N}(M^{-1}W^T(x-\mu),\, \sigma^2 M^{-1})$ —— 精確的 soft encoding。
   - ML 解 $\sigma^2_{ML} = \frac{1}{D-M}\sum_{i=M+1}^D \lambda_i$ —— 被丟掉的特徵值的平均就是噪聲。

3. **Standard PCA 是 $\sigma^2 \to 0$ 的特例**：PPCA 多了厚度（噪聲），多了機率，但少了那份幾何簡潔。

4. **PPCA 到 VAE 的關鍵跨越**：把線性 decoder 換成 nonlinear network，後驗從 closed-form 變成 intractable，這迫使 VAE 必須引入 encoder 網路做 approximate inference。Closed-form 的代價，正是 PPCA 表達能力的天花板。

5. **EM 的角色**：PPCA 有 closed-form 所以不需要跑 EM，但 EM 的思路（E-step inference，M-step update）就是 VAE training loop 的原型，只是把精確的步驟全部換成近似版。
