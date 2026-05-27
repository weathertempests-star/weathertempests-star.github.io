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

# Linear Factor Models 與 Probabilistic PCA (PPCA)

## 這篇文章想回答什麼問題？

PCA 大多數人都學過，但通常是從線性代數的角度：找一組正交基底，讓投影後的資料變異數最大。

這篇文章想從另一個角度重新理解同一件事：

> 如果資料背後真的有一組我們看不見的 latent factors 在驅動，那這件事要怎麼用機率語言說清楚？只觀察到 $x$ 卻看不到 $z$，模型還能不能被學出來？

說到底，這篇文章想回答三個問題：

1. **Latent variable model 的生成故事是什麼？** 資料 $x$ 是怎麼從 $z$ 產生的？
2. **PPCA 和 Standard PCA 的本質差別在哪？** 為什麼「加上機率」不只是換個說法？
3. **為什麼 PPCA 是通往 VAE 的橋樑？** linear decoder 換成 nonlinear 之後，什麼東西開始壞掉？

如果把這篇濃縮成一句話：

> PPCA 用一個最簡單、但仍然完整的機率生成模型，把 PCA 從幾何方法提升成可以做推論、做學習、做生成的 probabilistic model；而它真正的價值，是讓你看清楚 VAE 到底在改什麼。

## 摘要

- PPCA 是在 PCA 之上加入機率化（噪聲模型與 likelihood），因此能做推論與生成。
- 在線性 decoder 的情況下，PPCA 的 posterior 與 ML 解都有封閉解；把 decoder 換成非線性（例如神經網路）後，就引出 VAE 的必要性。
- EM 的思路（E-step 做 inference，M-step 更新參數）是理解更複雜潛變數模型的重要橋樑。


---

## 為什麼 Standard PCA 不夠用？

### PCA 做的事

PCA 的核心是找一個低維子空間，讓資料投影進去之後保留最多的變異數。具體來說，它找到一組正交向量 $w_1, \dots, w_M$，讓：

$$
\text{Var}\left(w_i^T x\right) \text{ 盡量大}
$$

結果就是：資料的主要變化方向被抓到了，次要的 noise 被丟掉了。

### PCA 做不到的事

但 PCA 有幾件事做不到：

**第一，它沒有噪聲模型。** PCA 只知道資料「大約落在某個超平面上」，但說不清楚偏離那個平面的部分是噪聲還是訊號。沒有噪聲模型，就沒辦法做推論——你無法說「給定觀測 $x$，latent code $z$ 的分佈是什麼」。

**第二，它不能生成新樣本。** 你可以把 $x$ 投影到 $z$，但如果我給你一個 $z$，你怎麼生成一個「合理的」$x$？PCA 是 deterministic 的，這個問題沒有清楚答案。

**第三，它沒有 likelihood。** 你沒辦法問「這筆資料符合這個模型的機率是多少」，因為根本沒有定義機率。

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

PPCA 把這個故事用 Gaussian 寫成公式。

**Latent prior：**

$$
p(z) = \mathcal{N}(z;\, 0,\, I)
$$

$z \in \mathbb{R}^M$ 是低維的，每個維度獨立標準高斯。

**生成過程（Decoder）：**

$$
p(x \mid z) = \mathcal{N}(x;\, Wz + \mu,\, \sigma^2 I)
$$

$x \in \mathbb{R}^D$ 是高維的觀測。給定 $z$，$x$ 的均值是 $Wz + \mu$（線性映射加平移），covariance 是 isotropic 的 $\sigma^2 I$。

用方程式寫出生成過程：

$$
x = Wz + \mu + \epsilon, \qquad \epsilon \sim \mathcal{N}(0,\, \sigma^2 I)
$$

這個式子說的事情很直觀：**資料 = 低維 latent code 的線性投影 + 平均位移 + 隨機噪聲**。

> **Note：** 這裡 $M < D$，也就是 latent dimension 比 observation dimension 小。如果 $M = D$ 且 $W$ 可逆，模型就退化成一般的 Gaussian，沒有降維效果。

### 幾何直覺

想像資料大致落在一個超平面附近：

- 超平面由 $Wz + \mu$ 的像（image）定義，它是一個 $M$ 維的 affine subspace 嵌在 $D$ 維空間裡。
- $\sigma^2$ 控制資料點離超平面的厚度。
- $\sigma^2 \to 0$ 時，資料幾乎精確落在超平面上，就退化回 standard PCA 的幾何圖像。

所以 PPCA 不是一個「平面」，而是一個「有厚度的高機率流形」，有人稱它為 pancake-shaped distribution。

---

## 數學推導

### 邊際分佈 $p(x)$

從生成過程出發，我們可以把 $z$ 積分掉，得到 $x$ 的邊際分佈。

因為 $z$ 和 $\epsilon$ 都是 Gaussian 且相互獨立，線性組合後仍然是 Gaussian：

$$
p(x) = \int p(x \mid z)\, p(z)\, dz = \mathcal{N}(x;\, \mu,\, C)
$$

**計算 $C$：** 用 $x - \mu = Wz + \epsilon$ 代入

$$
C = \mathrm{Cov}(x) = \mathbb{E}[(Wz + \epsilon)(Wz + \epsilon)^T]
$$

展開後，由於 $z$ 和 $\epsilon$ 獨立且均值為零，交叉項消失：

$$
C = W \underbrace{\mathbb{E}[zz^T]}_{=I} W^T + \underbrace{\mathbb{E}[\epsilon\epsilon^T]}_{=\sigma^2 I} = WW^T + \sigma^2 I
$$

這個 $C$ 是模型能接觸到的 covariance，它把 signal（$WW^T$）和 noise（$\sigma^2 I$）分開了。

### 後驗分佈 $p(z \mid x)$

這是做 inference 的方向：給定觀測 $x$，latent $z$ 的後驗分佈是什麼？

用 Bayes rule：$p(z \mid x) \propto p(z)\, p(x \mid z)$

把兩個 Gaussian 乘起來，指數部分整理 $z$ 的二次型：

$$
\log p(z \mid x) \propto -\frac{1}{2}z^T z - \frac{1}{2\sigma^2}(x - \mu - Wz)^T(x - \mu - Wz)
$$

展開第二項並整理 $z$ 相關的項，定義：

$$
M = W^T W + \sigma^2 I \in \mathbb{R}^{M \times M}
$$

完成配方後得到：

$$
\boxed{p(z \mid x) = \mathcal{N}\!\left(z;\; M^{-1}W^T(x - \mu),\; \sigma^2 M^{-1}\right)}
$$

**後驗均值的意義：**

$$
\mathbb{E}[z \mid x] = M^{-1}W^T(x - \mu)
$$

這是 $x$ 的 soft encoding，可以想成 probabilistic 版的 PCA projection。注意它不是 $W^T(x-\mu)$ 那樣簡單的投影，而是經過 $M^{-1}$ 修正過的，這個修正考慮了 latent dimensions 之間的相關性以及噪聲的影響。

**後驗 covariance 的意義：**

$$
\mathrm{Cov}(z \mid x) = \sigma^2 M^{-1}
$$

噪聲 $\sigma^2$ 越大，後驗越不確定（covariance 越大），這很符合直覺：如果觀測噪聲很大，你對 latent 的推斷自然就越不確定。

> **Insight：** $\sigma^2 \to 0$ 時，$M^{-1} \to (W^T W)^{-1}$，後驗均值趨近於 $(W^T W)^{-1} W^T (x - \mu)$，也就是 standard PCA 的 orthogonal projection。PPCA 在這個極限下回到幾何方法，但在 $\sigma^2 > 0$ 時它多了「不確定性」的描述。

---

## 參數學習

### 目標：Maximum Likelihood

有 $N$ 筆觀測 $\{x_n\}$，沒有 $z_n$。把 $z$ 積分掉後，每筆資料的 log-likelihood 是：

$$
\log p(x_n;\, W, \mu, \sigma^2) = \log \mathcal{N}(x_n;\, \mu,\, C)
= -\frac{D}{2}\log(2\pi) - \frac{1}{2}\log|C| - \frac{1}{2}(x_n - \mu)^T C^{-1}(x_n - \mu)
$$

對整個資料集：

$$
\log p(X;\, W, \mu, \sigma^2) = -\frac{ND}{2}\log(2\pi) - \frac{N}{2}\log|C| - \frac{1}{2}\sum_{n=1}^N (x_n - \mu)^T C^{-1}(x_n - \mu)
$$

### Closed-Form 解

先解 $\mu$：對稱結構下最優解就是樣本均值

$$
\mu_{ML} = \bar{x} = \frac{1}{N}\sum_{n=1}^N x_n
$$

再解 $W$ 和 $\sigma^2$：定義樣本 covariance matrix

$$
S = \frac{1}{N}\sum_{n=1}^N (x_n - \bar{x})(x_n - \bar{x})^T
$$

對 $S$ 做 eigendecomposition：$S = U \Lambda U^T$，其中 $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_D$。

ML 解為：

$$
\boxed{W_{ML} = U_M\,(\Lambda_M - \sigma^2_{ML}\, I)^{1/2}\, R}
$$

$$
\boxed{\sigma^2_{ML} = \frac{1}{D - M}\sum_{i=M+1}^{D} \lambda_i}
$$

其中：
- $U_M$：前 $M$ 個 principal eigenvectors
- $\Lambda_M = \mathrm{diag}(\lambda_1, \dots, \lambda_M)$
- $R$：任意 orthogonal matrix（旋轉不唯一性，後面會說）

**$\sigma^2_{ML}$ 的幾何意義：** 它是「沒有被前 $M$ 個主成分解釋掉的特徵值的平均」，也就是剩餘 noise 的平均水準。若 $M = D-1$，則 $\sigma^2_{ML} = \lambda_D$；若 $M \to D$，$\sigma^2 \to 0$。

### 旋轉不唯一性

把 $W$ 換成 $WR$（$R$ 是任意 orthogonal matrix），covariance 不變：

$$
(WR)(WR)^T = W R R^T W^T = WW^T
$$

因此 $p(x)$ 完全不受影響。**模型識別的不是 latent axes 的方向，而是它們張成的 subspace。** 這和 standard PCA 一樣，eigenvectors 本身可以任意旋轉而不改變投影子空間。

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
