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

# Autoencoder 與 Variational Autoencoder：從壓縮到生成

## 這篇文章想回答什麼問題？

如果你只是想壓縮資料、再重建回來，普通的 autoencoder 就夠了。但如果你想要一個可以**憑空生成**新樣本的模型，問題就變得更根本。

這篇文章想釐清的問題鏈大概長這樣：

1. Autoencoder 做得很好，為什麼我們還需要 VAE？  
2. 生成模型的核心困難是什麼——為什麼 posterior 幾乎不可能直接算出來？  
3. 怎麼用「近似分佈」繞過這個困難，又怎麼把訓練目標寫成封閉的數學形式？  
4. 「抽樣」這件事為什麼會卡住 backpropagation，Reparameterization Trick 是如何解套的？  
5. CVAE 加了什麼、解決了什麼、又特別適合哪類任務？
6. 以 *Conditional VAE for Video Prediction* 為例，CVAE 在實際應用中長什麼樣子？

> Note：本文記號使用 $\phi$ 代表 encoder 參數，$\theta$ 代表 decoder 參數，跟主流論文一致。原始課堂使用 $\theta'$ 代表 encoder、$\theta$ 代表 decoder，閱讀時注意對應即可。

在系列位置上，這篇是承接上一章 PPCA 的下一步：PPCA 還能把 posterior 精確算出來，但一旦 decoder 換成神經網路，這個封閉性就消失了。VAE 要處理的，正是「posterior 算不出來之後，怎麼還能把生成模型訓練起來」這件事。

如果把整篇文章收斂成一條閱讀路線，那會是：先看 Autoencoder 的盲點，再把 VAE 寫成 latent variable model，接著用變分推論和 ELBO 解決 posterior 不可積分的問題，最後把 reparameterization、CVAE 與 video prediction 串成一個完整應用。

## 摘要

- Autoencoder 適合壓縮與重建，但 latent space 無機率結構，無法直接生成新樣本。
- VAE 引入 encoder 作為近似 posterior，並以 ELBO（reconstruction + KL）同時學習 decoder 與 encoder，使 latent space 成為可採樣的機率空間。
- Reparameterization trick 讓隨機採樣節點可微；CVAE 則把 condition 加入 encoder/decoder，適合一對多的生成任務。

---

## 一、先把問題說清楚：Autoencoder 的強項與盲點

一般 autoencoder 的邏輯很直白：

$$
x \xrightarrow{\text{encoder}} z \xrightarrow{\text{decoder}} \hat{x}
$$

訓練目標是讓重建誤差 $\|x - \hat{x}\|^2$ 盡量小。這個設計已經很有用——它能學到緊湊的 latent representation，拿來做降維、去噪或異常偵測都沒問題。

但如果你想做的是**生成**，autoencoder 就遇到了一個它從沒被設計來解決的問題：

> latent code $z$ 只是一個壓縮後的向量，不代表它來自某個有結構的機率空間。

舉個例子：你拿一張狗的圖片跑過 encoder，得到 $z = [0.3, -1.2, 0.8, \ldots]$。這個向量對 decoder 有意義，因為它是真實圖片壓縮出來的。但如果你隨手抽一個 $z' = [1.1, 0.4, -0.3, \ldots]$，decoder 拿到它幾乎肯定會輸出垃圾——因為 latent space 裡沒有任何規範說哪些位置是「有意義的」。

**VAE 要修的就是這件事：讓 latent space 本身成為一個有結構的機率分佈，訓練完之後可以直接從 prior 抽樣，再用 decoder 生成有意義的新樣本。**

更精確地說，普通 autoencoder 只有 deterministic mapping $x \mapsto z \mapsto \hat{x}$，沒有明確的 $p(z)$、$p(x \mid z)$ 與 $p(z \mid x)$。VAE 要做的，是把「壓縮」升級成「推論」，把「重建」升級成「生成」。

---

## 二、VAE 的生成故事：先把模型寫出來

VAE 是一個 **latent variable model**。它假設資料的生成過程是這樣的：

```
p(z) → p(x | z) → 觀測到 x
```

用數學寫：

$$
p(z) = \mathcal{N}(z;\, 0,\, I)
$$

$$
p_\theta(x \mid z) = \mathcal{N}\!\left(x;\, f_\theta(z),\, \sigma_x^2 I\right)
$$

其中 $f_\theta$ 是 decoder 神經網路，輸出 $x$ 的均值。

所以整個模型可以簡單寫成一個 joint distribution：

$$
p_\theta(x, z) = p(z)\, p_\theta(x \mid z)
$$

方向很清楚：先從 prior 抽 $z$，再透過 decoder 得到 $x$。**但訓練時，我們只有 $x$，沒有 $z$。** 這帶出了整個框架最核心的困難。

這裡和 PPCA 的語法完全一致，只是把線性的 decoder 換成了非線性的 $f_\theta$。因此可以把 VAE 看成 linear latent variable model 的一般化版本，只是這個一般化會讓 posterior 失去 closed form。

---

## 三、為什麼 Posterior 不能直接算？

訓練時我們真正需要的是 posterior $p_\theta(z \mid x)$——給定觀測 $x$，latent code 的分佈應該長什麼樣子。根據 Bayes rule：

$$
p_\theta(z \mid x) = \frac{p(z)\, p_\theta(x \mid z)}{p_\theta(x)}, \qquad p_\theta(x) = \int p(z)\, p_\theta(x \mid z)\, dz
$$

問題出在分母。這個積分涵蓋整個 latent space，而 $f_\theta(z)$ 是 nonlinear 神經網路，所以 integrand 不是 Gaussian，沒有 closed form。

用一個 scalar 版本直觀看一下：

$$
\log p_\theta(z \mid x) \propto -\frac{1}{2}z^2 - \frac{1}{2\sigma^2}\left(x - f_\theta(z)\right)^2
$$

如果 $f_\theta(z)$ 是線性的，第二項還是 $z$ 的二次式，整體仍然是 Gaussian——這就是為什麼 PPCA 可以精確推論。一旦 $f_\theta$ 變成 nonlinear neural network，整個 exponent 就不再是簡單的二次型，posterior 也不再有解析解。

> Insight：這正是 PPCA 和 VAE 的分界線。PPCA 可以精確推論，是因為 decoder 還在線性世界；VAE 把 decoder 升級為神經網路，換來更強的生成能力，也換來了 approximate inference 的代價。

從技術上看，困難不是 sampling 本身，而是 $p_\theta(z \mid x)$ 在 nonlinear decoder 下不再屬於容易處理的分佈族，所以我們才需要用一個參數化分佈去逼近它。

### 線性 decoder 時，posterior 為什麼能算出來？

如果把 decoder 寫成線性形式

$$
f_\theta(z) = Wz + b
$$

並且仍然假設

$$
p(z) = \mathcal{N}(z; 0, I), \qquad p_\theta(x \mid z) = \mathcal{N}(x; Wz + b, \sigma_x^2 I)
$$

那麼 posterior 的 log-density 可以寫成

$$
\log p_\theta(z \mid x)
= -\frac{1}{2} z^\top z - \frac{1}{2\sigma_x^2} \lVert x - Wz - b \rVert^2 + \text{const}
$$

把平方項展開：

$$
\lVert x - Wz - b \rVert^2
= (x-b)^\top(x-b) - 2 z^\top W^\top(x-b) + z^\top W^\top W z
$$

代回去之後，所有和 $z$ 有關的項都只到二次：

$$
\log p_\theta(z \mid x)
= -\frac{1}{2} z^\top \left(I + \frac{1}{\sigma_x^2} W^\top W\right) z
+ \frac{1}{\sigma_x^2} z^\top W^\top(x-b) + \text{const}
$$

因此 posterior 仍然是 Gaussian。這正是 PPCA 可以精確推論的原因；一旦 $Wz+b$ 換成 nonlinear $f_\theta(z)$，上述二次型結構就消失了。

---

## 四、變分推論：用 $q_\phi(z \mid x)$ 近似 Posterior

既然 posterior 算不出來，我們就用一個可學習的近似分佈來取代它：

$$
q_\phi(z \mid x) \approx p_\theta(z \mid x)
$$

這就是 VAE encoder 的真正角色——不是「壓縮」，而是在做 **approximate posterior inference**。

$q_\phi$ 之所以要依賴 $x$，是因為不同的資料點，其 posterior 本來就不一樣。如果不看 $x$ 直接用 prior $p(z)$，那根本沒有在做推論。

最常見的設計是讓 encoder 輸出 Gaussian 的參數：

$$
q_\phi(z \mid x) = \mathcal{N}\!\left(z;\, \mu_\phi(x),\, \operatorname{diag}(\sigma_\phi^2(x))\right)
$$

也就是，encoder 看到 $x$ 後輸出兩組向量：均值 $\mu_\phi(x)$ 和方差 $\sigma_\phi^2(x)$。這樣做有兩個好處：形式簡單、KL divergence 有封閉解（後面會算）。

更精確地說，我們不是在逼近「一個點」，而是在逼近「每個資料點對應的一個條件分佈」。這也是 VAE 跟普通 autoencoder 最根本的不同：前者學的是 posterior family，後者只學一個 deterministic code。

---

## 五、ELBO：把 log-likelihood 拆成可訓練的目標

VAE 的訓練目標從 $\log p_\theta(x)$ 出發。插入 $q_\phi(z \mid x)$ 並用 Jensen inequality：

$$
\log p_\theta(x)
= \log \int q_\phi(z \mid x)\frac{p_\theta(x,z)}{q_\phi(z \mid x)}\,dz
\;\ge\;
\underbrace{\mathbb{E}_{q_\phi(z \mid x)}\!\left[\log \frac{p_\theta(x,z)}{q_\phi(z \mid x)}\right]}_{\mathcal{L}_{\text{ELBO}}(x)}
$$

把 joint 拆開 $\log p_\theta(x,z) = \log p_\theta(x \mid z) + \log p(z)$，ELBO 整理成：

$$
\boxed{
\mathcal{L}_{\text{ELBO}}(x)
=
\underbrace{\mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x \mid z)\right]}_{\text{Reconstruction Term}}
-
\underbrace{\mathrm{KL}\!\left(q_\phi(z \mid x)\,\|\,p(z)\right)}_{\text{Regularization Term}}
}
$$

更精確地說，兩者的關係是：

$$
\log p_\theta(x) = \mathcal{L}_{\text{ELBO}}(x) + \mathrm{KL}\!\left(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\right)
$$
z = g_\phi(x, \epsilon), \qquad g_\phi(x, \epsilon) = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
$$

把這個式子再拆一次，可以看得更清楚：

$$
\log p_\theta(x) - \mathcal{L}_{\text{ELBO}}(x)
=
\mathrm{KL}\!\left(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\right)
z \sim q_\phi(z \mid x) = \mathcal{N}\!\left(\mu_\phi(x),\, \sigma_\phi^2(x)\right)
$$
所以最小化 negative ELBO 的本質，就是同時做兩件事：一方面讓 decoder 重建得好，另一方面讓 approximate posterior 不要偏離真正的 posterior 太遠。

### 兩項各自在做什麼？

**Reconstruction Term**：假設 decoder 是 Gaussian，展開後得到：

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$
$$

最大化這一項，等價於最小化重建誤差（squared error）。

如果把觀測模型改成 Bernoulli，這一項就會對應 binary cross-entropy；如果是連續值影像，Gaussian 假設通常就導向 MSE。也就是說，重建項的形式不是任意選的，而是由 $p_\theta(x \mid z)$ 的分佈假設決定。

**Regularization Term**：把 encoder 的 posterior 拉向 prior，確保 latent space 的幾何結構是有意義的。如果沒有這一項，encoder 可以把每個 $x$ 壓成 delta function（完全沒有不確定性），重建 loss 最小，但 latent space 會四分五裂，根本沒辦法採樣生成。

> Insight：可以把 latent space 想成「一把大雨傘裡有很多小雨傘」。$p(z)$ 是大傘，每個 $q_\phi(z \mid x_i)$ 是小傘。KL regularization 讓所有小傘都收在大傘底下。訓練完之後，從大傘（prior）隨手抽一個點，decoder 就能生成有意義的樣本。

這也解釋了為什麼 VAE 的 latent space 通常比普通 autoencoder 更適合插值與採樣：ELBO 不只約束重建，還把 latent 的幾何形狀拉回到一個可採樣的機率空間。

---

## 六、KL Divergence 的封閉解

兩個 Gaussian 之間的 KL divergence 有漂亮的 closed form。

更一般地，如果 $q = \mathcal{N}(\mu_q, \Sigma_q)$、$p = \mathcal{N}(0, I)$，那麼

$$
\mathrm{KL}(q \,\|\, p)
= \frac{1}{2}
\left(
\operatorname{tr}(\Sigma_q) + \mu_q^\top \mu_q - d - \log \det \Sigma_q
\right)
$$

當 $\Sigma_q$ 是 diagonal matrix 時，就退化成逐維相加的形式。令 $q = \mathcal{N}(\mu_q, \operatorname{diag}(\sigma_q^2))$、$p = \mathcal{N}(0, I)$（標準 Gaussian prior），則：

$$
\mathrm{KL}(q \,\|\, p)
= \frac{1}{2}\sum_{j=1}^{d}
\left(
\mu_{q,j}^2 + \sigma_{q,j}^2 - \log \sigma_{q,j}^2 - 1
\right)
$$

這個 closed form 是 VAE 能 end-to-end 訓練的關鍵。整個 ELBO 裡，reconstruction term 透過 Monte Carlo 近似，regularization term 精確計算，加在一起就是可以直接對 $\phi$ 和 $\theta$ 求梯度的 loss。

如果 prior 不只是標準 Gaussian，而是一般的 $\mathcal{N}(\mu_p, \Sigma_p)$，KL 仍然能寫成 closed form，只是會多出 trace 與 log-determinant 項。真正讓訓練變麻煩的，是更複雜的 prior family，而不是單純把均值和平移改掉。

> Warning：一旦換成更複雜的 prior（例如 Gaussian mixture），KL 通常就沒有封閉解，訓練難度大幅上升。Gaussian 之所以是預設選擇，正是因為它在表達能力和可解析性之間找到了一個很好的平衡點。

---

## 七、Reparameterization Trick：讓隨機節點可以反傳梯度

到這一步還有最後一個問題：**sampling 不可微**。

如果你直接寫：

$$
z \sim q_\phi(z \mid x) = \mathcal{N}\!\left(\mu_\phi(x),\, \sigma_\phi^2(x)\right)
$$

梯度就會卡在這個隨機節點，沒辦法傳回 encoder。

Reparameterization Trick 的做法，是把 randomness 從 $z$ 挪到一個跟參數無關的 $\epsilon$：

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

等價地，也可以寫成

$$
z = g_\phi(x, \epsilon), \qquad g_\phi(x, \epsilon) = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
$$

這樣一來，隨機性只存在於 $\epsilon$，而 $\phi$ 只負責定義把標準噪聲映射到 posterior sample 的確定性變換。對任意可微函數 $f$，梯度就可以寫成：

$$
\nabla_\phi\, \mathbb{E}_{q_\phi(z \mid x)}[f(z)]
= \nabla_\phi\, \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}
\!\left[f\!\left(g_\phi(x, \epsilon)\right)\right]
= \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}
\!\left[\nabla_z f\!\left(g_\phi(x, \epsilon)\right)\, \nabla_\phi g_\phi(x, \epsilon)\right]
$$

右邊的梯度就能直接穿過 $\mu_\phi$ 和 $\sigma_\phi$ 兩個神經網路，backpropagation 正常運作。

這一步的核心，是把「隨機節點」改寫成「確定性函數 + 外部噪聲」。換句話說，參數不再直接控制 sampling，而是控制一個把標準噪聲映射到 posterior sample 的變換。

整個 VAE 的資料流整理如下：

```
Input x                         [B, D]
         │
         ├─ Encoder q_phi ──> mu      [B, K]
         │                └─> logvar  [B, K]
         │
epsilon ~ N(0, I)               [B, K]
         │
         └─ z = mu + exp(0.5 * logvar) ⊙ epsilon    [B, K]
                   │
                   └─ Decoder p_theta ──> x_hat      [B, D]

Loss = Reconstruction Loss + KL Term
```
 
## 實作提示

- 在圖像任務中，`p_\theta(x|z)` 常用 Bernoulli（binary）或 Gaussian（continuous），對應的 reconstruction loss 分別是 binary cross-entropy 或 MSE。
- 訓練初期可以採用 KL annealing（或逐步增加 `beta`）來減緩 posterior collapse；若仍發生 collapse，可調低 learning rate、增加 decoder capacity 或嘗試更強的 posterior expressiveness（如 flow-based posterior）。
- 對於 `logvar` 的數值穩定化（clamp 範圍或小常數）能避免標準差為 0 或數值爆炸。

這三點剛好對應到數學面：Gaussian decoder 決定 reconstruction term 的形式，KL annealing 是在控制正則化強度，而 `logvar` 的穩定化則是在避免 reparameterization 時的數值病態。

---

## 八、訓練完之後，Encoder 可以丟掉

這是 VAE 很常被忽略、但其實很重要的地方。

**訓練期間**，encoder 的角色是幫你從資料 $x$ 推出 approximate posterior，讓整個 ELBO 有辦法計算和優化。

**生成期間**，你不需要 encoder。流程是：

1. 從 prior $p(z) = \mathcal{N}(0, I)$ 抽一個 $z$
2. 把 $z$ 丟給 decoder $p_\theta(x \mid z)$
3. 得到新生成的樣本

Encoder 是訓練的必要手段，decoder 才是最後的生成器。

---

## 九、Conditional VAE：加上一個 Conditioning Signal

如果輸入和輸出之間是 **one-to-many mapping**，普通的 VAE 就不夠了。

想像你手上有一張被遮住的鳥圖，想補全被遮的部分。被遮的部分可能有很多種合理的補法——羽毛的顏色、角度、背景，都可以有不同的版本。這是一個典型的一對多問題。

如果用 deterministic CNN 搭配 MSE loss 訓練，它會傾向學 conditional mean：

$$
f^*(c) = \arg\min_f \mathbb{E}\!\left[\|x - f(c)\|^2\right] = \mathbb{E}[x \mid c]
$$

當 $p(x \mid c)$ 是 multimodal 的時候，conditional mean 往往落在幾個 mode 的中間——視覺上就變成輪廓模糊、細節被平均掉的圖。

**CVAE 的解法**是讓 latent variable $z$ 承載那個不確定性。給定 conditioning signal $c$，模型學的是整個條件分佈 $p_\theta(x \mid z, c)$，而不是一個點估計。

CVAE 的架構只需要把 $c$ 一起送進 encoder 和 decoder：

$$
q_\phi(z \mid x, c), \qquad p_\theta(x \mid z, c), \qquad p(z)
$$

對應的 ELBO：

$$
\mathcal{L}_{\text{CVAE}}(x, c)
=
\mathbb{E}_{q_\phi(z \mid x,c)}\!\left[\log p_\theta(x \mid z, c)\right]
- \mathrm{KL}\!\left(q_\phi(z \mid x, c)\,\|\,p(z)\right)
$$

生成時：先從 prior 抽 $z$，再給定同一個 $c$ 生成不同的合理輸出。同一個 condition 搭配不同的 $z$，就能得到多樣化的生成結果。

> Insight：CVAE 不是把不確定性消掉，而是讓模型學會「在給定 $c$ 的前提下，哪些變化是合理的」，再用 $z$ 去採樣那個合理的變化空間。

如果把 $c$ 視為已知條件，CVAE 實際上是在學條件分佈 $p_\theta(x \mid c)$ 的 latent factorization：condition 決定主方向，latent 決定在這個主方向底下仍然合理的變化。

---

## 十、實例：Conditional VAE for Video Prediction

現在用一個具體的應用把所有概念串起來：**用 CVAE 預測影片的下一幀（甚至多幀）**。

### 問題設定

給定一段影片的前 $T$ 幀 $x_{1:T}$，預測接下來的 $K$ 幀 $x_{T+1:T+K}$。

如果只看單一步的下一幀，其實要建模的是條件分佈

$$
p_\theta(x_{T+1} \mid c_T)
= \int p_\theta(x_{T+1} \mid z, c_T)\, p_\theta(z \mid c_T)\, dz
$$

這個積分和前面的 VAE 完全同構，只是把「資料本身」換成「在 context 下的未來」。也因此，影片預測的難點不是把畫面壓縮，而是要讓 latent $z$ 去承載那些只靠歷史 context 還不足以決定的未來變化。

這個任務的核心困難正是一對多：就算前幾幀完全一樣，未來的發展也可能不同。一個人舉起手，接下來可能揮手、可能拿東西，也可能放下來。如果用 deterministic model 預測，得到的只會是幾種可能的模糊平均。

代表性的工作是 **Stochastic Video Generation (SVG)**（Denton & Fergus, 2018），以及後續的多種 CVAE-based video prediction 架構。下面用這個框架來說明。

### 模型架構

整個模型由三個部分組成：

**Frame Encoder**：用 CNN 把每一幀壓成特徵向量。

$$
h_t = \text{CNN}_\phi(x_t)
$$

**Temporal Model（通常是 LSTM）**：沿時間軸整合歷史，輸出一個 context vector $c_t$，代表「到目前為止發生了什麼」。

$$
c_t = \text{LSTM}(h_1, h_2, \ldots, h_t)
$$

**CVAE（核心）**：

- Prior network：給定當前 context $c_T$，輸出 prior 的 $\mu_p, \sigma_p$。
- Posterior network（訓練時用）：給定 context $c_T$ 和真實的下一幀 $x_{T+1}$，輸出 posterior 的 $\mu_q, \sigma_q$。
- Decoder：給定 $z$ 和 $c_T$，生成預測幀 $\hat{x}_{T+1}$。

這三個模組的分工很關鍵：CNN 把像素壓成 frame-level feature，LSTM 把 feature 壓成歷史 context，CVAE 再把「context + latent」映射成未來的條件分佈。真正負責多樣性的是 latent $z$，不是 CNN 或 LSTM。

### 為什麼需要一個 Prior Network，而不是直接用標準 Gaussian？

這是 video prediction CVAE 的關鍵設計選擇。

標準 VAE 使用固定的 $p(z) = \mathcal{N}(0, I)$ 作為 prior。但在 video prediction 裡，不同的 context $c_T$ 對應的「合理未來」分佈是不同的。

例如：一個人正在跑步的 context，$z$ 應該傾向採樣「往前的動作」；而一個人靜止站立的 context，$z$ 可以採樣更多樣化的後續動作。

所以我們讓 prior 也依賴 context：

$$
p_\theta(z \mid c_T) = \mathcal{N}\!\left(\mu_p(c_T),\, \sigma_p^2(c_T)\right)
$$

從變分推論的角度，這代表我們要最大化的是條件對數似然 $\log p_\theta(x_{T+1} \mid c_T)$。插入 posterior $q_\phi(z \mid x_{T+1}, c_T)$ 後，再套 Jensen inequality，就得到

$$
\log p_\theta(x_{T+1} \mid c_T)
= \log \int q_\phi(z \mid x_{T+1}, c_T)
\frac{p_\theta(x_{T+1} \mid z, c_T)\, p_\theta(z \mid c_T)}{q_\phi(z \mid x_{T+1}, c_T)}\, dz
\ge \mathcal{L}
$$

這樣 ELBO 就改寫成：

$$
\mathcal{L}
=
\mathbb{E}_{q_\phi(z \mid x_{T+1}, c_T)}\!\left[\log p_\theta(x_{T+1} \mid z, c_T)\right]
-
\mathrm{KL}\!\left(q_\phi(z \mid x_{T+1}, c_T)\,\|\,p_\theta(z \mid c_T)\right)
$$

這個式子和標準 VAE 的差別，只在於 prior 也被條件化了。訓練時，posterior network 可以看到真實未來；推論時，只有 prior network 能用，所以 KL 項同時扮演 regularizer 和 train/test bridge 的角色。

### 訓練 vs. 推論：Train/Test 的不對稱性

| | 訓練階段 | 推論階段 |
|---|---|---|
| $z$ 的來源 | Posterior network：看過 $x_{T+1}$，知道真實未來 | Prior network：只看 context $c_T$，不知道未來 |
| 目的 | 讓模型學會「給定未來，應該用什麼 $z$」 | 讓模型學會「從 context 猜合理的 $z$ 分佈，再生成」 |
| 訓練技巧 | KL 項把 posterior 拉向 prior，縮小 train/test gap | — |

這個不對稱性是所有 CVAE 應用都必須面對的核心問題。KL term 的作用不只是正則化，更是強迫 posterior 不要和 prior 差太遠，讓推論時的 prior 採樣也能生成合理的結果。

### 一次完整的資料流

**訓練時（有 ground truth $x_{T+1}$）：**

```
x_1, ..., x_T  ─[CNN + LSTM]─>  context c_T       [B, d_c]

Prior network(c_T)     ──>  mu_p, sigma_p           [B, K]
Posterior network(c_T, x_{T+1}) ──>  mu_q, sigma_q [B, K]

epsilon ~ N(0, I)                                   [B, K]
z = mu_q + sigma_q ⊙ epsilon                        [B, K]

Decoder(z, c_T)  ──>  x_hat_{T+1}                  [B, C, H, W]

Loss = Reconstruction(x_{T+1}, x_hat_{T+1})
     + KL(N(mu_q, sigma_q) || N(mu_p, sigma_p))
```

**推論時（沒有 ground truth）：**

```
x_1, ..., x_T  ─[CNN + LSTM]─>  context c_T

Prior network(c_T)  ──>  mu_p, sigma_p

z ~ N(mu_p, sigma_p)    <── 從 prior 採樣，每次不同就生成不同的未來

Decoder(z, c_T)  ──>  x_hat_{T+1}
```

重複採樣不同的 $z$，就能看到從同一段歷史出發的多種可能未來。

這裡的模型細節很重要：如果沒有 prior network，推論時就只能從固定的標準常態抽樣，生成結果會和上下文脫節；如果沒有 posterior network，訓練時又不知道該把哪些 latent 對齊到真實未來。video prediction 的 CVAE 正是靠這個不對稱設計，把多樣性和條件一致性同時保住。

### KL Annealing：訓練穩定性技巧

實作上有一個常見問題：訓練初期，如果 KL 項太強，模型會傾向把 $z$ 完全忽略（posterior 退化成 prior），只靠 context $c_T$ 做預測——這叫 **posterior collapse**。

解法是 **KL annealing**：訓練初期把 KL 項的權重 $\beta$ 設得很小，讓模型先學會用 $z$ 做有意義的表示，再逐漸提高 $\beta$：

$$
\mathcal{L}_\beta = \mathbb{E}_{q_\phi}\!\left[\log p_\theta(x \mid z, c)\right] - \beta \cdot \mathrm{KL}\!\left(q_\phi \,\|\, p_\theta\right)
$$

$\beta$ 從 0 線性增加到 1，通常在幾千個 training step 內完成。

> Warning：$\beta$ 最終要回到 1，否則就不再是真正的 ELBO。在 $\beta < 1$ 的階段，模型只是在做 warm-up，不是在最佳化正確的目標。

從模型角度看，KL annealing 的作用是先讓 decoder 學會真的使用 $z$，避免一開始就被強 KL 壓到忽略 latent；對 video prediction 而言，這點尤其重要，因為 temporal context 很強，模型很容易走捷徑。

### 評估多樣性：不只看重建品質

由於 video prediction 本質上是一對多問題，只看重建誤差不夠。常見的評估指標還包括：

- **SSIM / PSNR**：測量單次重建的像素品質。
- **FVD（Fréchet Video Distance）**：類似 FID，在 feature space 評估生成影片的分佈是否和真實分佈接近。
- **Diversity**：對同一個 context 採樣多次，計算不同生成結果之間的差異，用來衡量模型有沒有真的捕捉到一對多的多樣性。

一個好的 CVAE video prediction model 應該同時在 quality（每次生成要合理）和 diversity（不同採樣要夠多樣）兩個面向表現好。

如果只看 MSE，模型通常會傾向輸出平均未來，結果反而更模糊；這也是為什麼一對多任務要同時看分佈層面的指標，而不是只看單次重建誤差。

---

## 十一、VAE、CVAE 與 GAN 的分岔點

生成模型有幾條主要的技術路線，了解它們的差異，才能知道什麼場景用什麼工具。

| | VAE | GAN | Diffusion |
|---|---|---|---|
| 訓練目標 | ELBO（likelihood lower bound）| Adversarial game | Denoising score matching |
| 樣本多樣性 | 靠 prior 採樣，多樣性由 latent space 決定 | 多樣性依賴 discriminator 的引導 | 多步 denoising 過程自然帶出多樣性 |
| 訓練穩定性 | 穩定，但 posterior collapse 是風險 | 不穩定，mode collapse 常見 | 穩定，但推論慢 |
| 樣本品質 | 偏模糊（重建 loss 傾向 mean） | 銳利，但可能有 artifact | 高品質 |
| 有無 encoder | 有（inference network）| 通常沒有 | 通常沒有（除非是 DDIM inversion） |
| 是否支援 latent interpolation | 是，且 latent space 連續光滑 | 通常是 | 較難直接做 |

VAE 偏向「我知道生成分佈，我要準確最佳化它」，GAN 偏向「我讓兩個網路對抗，讓品質提升」，Diffusion 則是「我學習一個去噪過程，慢慢從噪聲中長出樣本」。

CVAE 特別適合的場景是：任務有明確的 condition（已知條件），又需要生成多樣化的輸出（一對多映射）。Video prediction、image inpainting、conditional image synthesis 都是典型例子。

---

## 總結

| 概念 | 核心作用 |
|---|---|
| Autoencoder | 壓縮與重建，但 latent space 無結構，不能生成 |
| VAE Encoder $q_\phi(z \mid x)$ | 做 approximate posterior inference，讓訓練有辦法進行 |
| VAE Decoder $p_\theta(x \mid z)$ | 生成器，訓練完後才是真正的主角 |
| ELBO | $\log p_\theta(x)$ 的可計算下界，由 reconstruction + KL 兩項組成 |
| KL Term | 讓 latent space 有結構，是 prior 採樣生成的保證 |
| Reparameterization Trick | 把採樣的隨機性外包給 $\epsilon$，讓梯度可以流回 encoder |
| CVAE | 加入 condition signal，讓生成可以被控制，適合一對多映射任務 |
| Video Prediction CVAE | Prior 依賴 context，Posterior 在訓練時看真實未來，KL Annealing 防止 posterior collapse |

一句話總結：

> VAE 把 autoencoder 改造成可生成的機率模型；CVAE 再加上 conditioning，讓模型學會「在給定條件下，什麼樣的未來是合理且多樣的」。Video prediction 正是這個想法的一個自然應用。

如果要繼續往下讀，下一步通常就是把 VAE/CVAE 和其他生成路線放在一起比較，尤其是 GAN 與 Diffusion 在樣本品質、多樣性與訓練穩定性上的取捨。
