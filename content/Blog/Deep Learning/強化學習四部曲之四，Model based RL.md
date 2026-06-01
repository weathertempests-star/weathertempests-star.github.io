---
type: Blog-post
tags:
  - Deep-Learning
  - note
  - Reinforcement-Learning
created: 2026-05-29
updated: 2026-05-29
authors: WeiTa
status: active
---

---

## 這篇文章想回答什麼問題？

前幾篇分別談了 Value-based RL 和 Policy-based RL。
這兩條路線雖然思路不同，但有一個共同的前提：

> Agent 不知道環境怎麼運作。它只能跑進去試，等著看結果。

這個「閉眼試錯」的方式在很多任務裡根本行不通。讓機器人在真實世界每摔一次才學一點，代價太高；讓自駕車在馬路上亂走直到出車禍再更新 policy，更不現實。

Model-based RL 給的答案是：**先學世界怎麼運作，再拿這個模型去想要怎麼做。**

這篇文章想回答四個問題：

1. 世界模型到底在學什麼？能學哪些形式？
2. 模型誤差會怎麼累積，為什麼 model-based 不是免費的午餐？
3. Dyna-Q 如何把「真實經驗」和「想像經驗」結合在一起？
4. Dreamer 和 MuZero 各自怎麼把 latent world model 推到極致？

如果把這篇濃縮成一句話：

> Model-based RL 是讓 agent 先建構一張世界地圖，再用地圖規劃路線；但地圖永遠不等於地形，知道邊界在哪裡，才能用得好。

---

## 為什麼 Model-free RL 不夠用？

回顧一下前兩篇的設定。不管是 Q-learning 還是 policy gradient，它們都假設：

$$
(s_t, a_t, r_{t+1}, s_{t+1}) \sim \text{真實環境}
$$

每一筆資料都要和真實環境互動才能拿到。這有幾個顯而易見的問題。

**樣本效率低落。** DQN 訓練 Atari 遊戲需要幾千萬步的環境互動，真實機器人每執行一次動作大約需要幾十毫秒，幾千萬步換算下來是幾百個小時。

**不能做長期規劃。** 如果你連「執行這個動作，下一個 state 會是什麼」都不知道，你就沒辦法在採取行動前預先推演後果。只能靠 value function 把長期回報壓縮成一個數字，但那是事後估計，不是事前規劃。

**無法利用先驗結構。** 物理系統有動力學方程，棋盤遊戲有明確規則，這些結構都是白送的資訊。Model-free 方法通常完全忽略它們，從零開始學。

> **Insight：** Model-free 的本質是把所有知識都壓進 $Q(s,a)$ 或 $\pi(a|s)$，讓 value 或 policy 隱含地記住環境的行為。Model-based 則選擇顯式地學出環境模型，再讓 planning 來完成剩下的事。

---

## 核心框架：什麼是 World Model？

完整的 model-based RL 框架至少包含三個元件：

- **World model**：預測環境動態，也就是 $\hat{s}_{t+1} = f_\phi(s_t, a_t)$ 和 $\hat{r}_{t+1} = g_\phi(s_t, a_t)$。
- **Planner / Policy**：根據 world model 決定怎麼行動。
- **Learning loop**：用真實互動更新 world model，再用更新後的模型改善 policy。

這三個元件的關係可以寫成：

```
真實環境互動
    ↓
蒐集 (s, a, r, s') 
    ↓
訓練 world model   ←────────────┐
    ↓                           │
用 model 做 rollout / planning  │
    ↓                           │
改善 policy                     │
    ↓                           │
和真實環境互動 ──────────────────┘
```

整個流程是一個交替迭代的過程：真實資料改善模型，模型改善決策，決策帶來更多真實資料。

---

## World Model 的四種形式

### 1. Transition model（前向動態）

最基本的形式：給定當前 state 和 action，預測下一個 state。

$$
\hat{s}_{t+1} = f_\phi(s_t, a_t)
$$

訓練目標就是對真實環境轉移做回歸：

$$
\mathcal{L}_{dyn} = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} \left[ \| s_{t+1} - f_\phi(s_t, a_t) \|^2 \right]
$$

有時預測 delta 比直接預測 $s_{t+1}$ 更穩，因為殘差通常比絕對值小、更易學：

$$
\hat{s}_{t+1} = s_t + f_\phi(s_t, a_t)
$$

這和 ResNet 的 skip connection 有異曲同工之妙：讓網路只學偏差，而不是整個映射。

### 2. Reward model

預測即時 reward：

$$
\hat{r}_{t+1} = g_\phi(s_t, a_t)
$$

$$
\mathcal{L}_{rew} = \mathbb{E}_{(s_t, a_t, r_{t+1}) \sim \mathcal{D}} \left[ (r_{t+1} - g_\phi(s_t, a_t))^2 \right]
$$

動態模型告訴你「會發生什麼」，reward model 告訴你「值不值得」。兩者合在一起，才能讓 agent 在 model 裡做完整的 rollout。

### 3. Inverse dynamics model

逆向問題：給定前後兩個 state，推斷中間採取了什麼 action。

$$
\hat{a}_t = h_\phi(s_t, s_{t+1})
$$

這看起來像是「知道了結果，反推原因」。它在 imitation learning 和 goal-conditioned planning 裡特別有用：你知道目標 state 是什麼，但不確定要採取什麼 action，就可以反查。

### 4. Latent World Model

前三種都在 observation 空間裡操作。如果輸入是高維影像，直接在像素空間做 transition 學習成本極高，而且很多像素資訊根本和 task 無關。

Latent world model 的思路是先壓縮，再推演：

$$
z_t = e_\phi(o_t) \qquad \text{(encoder)}
$$

$$
\hat{z}_{t+1} = f_\phi(z_t, a_t) \qquad \text{(latent transition)}
$$

$$
\hat{o}_t = d_\psi(z_t) \qquad \text{(decoder，按需使用)}
$$

Latent space 維度低、語意密度高，在這裡做 rollout 效率遠勝於在像素空間。

---

## 數學核心：Model Bias 與 Compound Error

這是理解 model-based RL 最容易被忽略、卻最關鍵的地方。

### 單步誤差

假設 world model 對單步轉移的誤差是 $\epsilon$：

$$
\| \hat{s}_{t+1} - s_{t+1} \| \leq \epsilon
$$

這個 $\epsilon$ 可能很小，感覺可以接受。

### 多步 rollout 的誤差累積

但 model-based RL 不只用一步預測。Planning 或 imagined rollout 常常需要展開 $H$ 步：

$$
\hat{s}_1, \hat{s}_2, \dots, \hat{s}_H
$$

每一步都在上一步的預測基礎上繼續預測。誤差會以大約線性或指數的方式累積。

更精確地說，假設 $f_\phi$ 的 Lipschitz 常數為 $L$（也就是它對輸入的敏感程度），那麼 $H$ 步後的誤差上界大約是：

$$
\| \hat{s}_H - s_H \| \leq \epsilon \cdot \frac{L^H - 1}{L - 1}
$$

這個式子可以由遞迴不等式直接看出來。若令 $e_t = \|\hat{s}_t - s_t\|$，且單步預測滿足

$$
e_{t+1} \le \epsilon + L e_t
$$

那麼反覆展開可得

$$
e_H \le \epsilon \sum_{k=0}^{H-1} L^k = \epsilon \cdot \frac{L^H - 1}{L - 1}
$$

所以 compound error 本質上不是「誤差有沒有」，而是「誤差會不會被模型本身放大」。若 $L < 1$，誤差還可能被壓住；若 $L > 1$，就會朝指數方向爆開。

當 $L > 1$（也就是模型對誤差有放大效應）時，這個上界是指數級的：

$$
\| \hat{s}_H - s_H \| = O(\epsilon \cdot L^H)
$$

這就是 **compound error** 問題：即使單步誤差很小，長期 rollout 仍然可能完全偏離真實軌跡。

> **Warning：** 這也是為什麼 model-based 不是免費的午餐。一個「看起來訓練得不錯」的 world model，在做 20 步的 imagined rollout 時，可能已經在一個根本不存在的 state 空間裡瞎走。

### 對 value 估計的影響

如果我們用 imagined rollout 來估計 $H$ 步的 return：

$$
\hat{G}_t = \sum_{k=0}^{H-1} \gamma^k \hat{r}_{t+k} + \gamma^H \hat{V}(\hat{s}_{t+H})
$$

這其實是在把真實的 Bellman backup 近似成「前 $H$ 步用 model 展開，尾端用 value function 截斷」：

$$
G_t = \sum_{k=0}^{H-1} \gamma^k r_{t+k} + \gamma^H V(s_{t+H})
$$

理想上若 model 完全正確，$\hat{G}_t$ 就會等於對應的 $H$ 步 return；但只要 $\hat{r}$ 或 $\hat{s}$ 有偏差，尾端的 $\hat{V}(\hat{s}_{t+H})$ 會把這些偏差再放大一次。這也是為什麼 imagined rollout 長度不能無限制增加。

其中每一個 $\hat{r}_{t+k}$ 和 $\hat{s}_{t+k}$ 都帶著預測誤差，而且這些誤差在 $\hat{V}$ 上還會再放大一次。所以 imagined return 和真實 return 之間的 bias，會隨著 $H$ 增加而惡化。

**實務上的處理方式：**
- 縮短 rollout 長度 $H$，犧牲遠見換穩定性。
- 使用 ensemble of models，用模型間的分歧度來估計不確定性，自動縮短不確定區域的 rollout。
- 在 model 不確定度高的 state 停下來，改用真實環境取樣。

---

## Dyna-Q：最優雅的基礎算法

Dyna-Q 是 Sutton 在 1991 年提出的框架，它用一個極其簡潔的思路把真實經驗和模型幻想組合在一起。

### 基本架構

Dyna-Q 的每一步做三件事：

**Step 1：真實互動，更新 Q**

從真實環境拿到一筆 transition $(s_t, a_t, r_{t+1}, s_{t+1})$，用標準 Q-learning 更新：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

**Step 2：用真實 transition 更新 world model**

$$
f_\phi \leftarrow \text{fit}(s_t, a_t, s_{t+1}), \qquad g_\phi \leftarrow \text{fit}(s_t, a_t, r_{t+1})
$$

**Step 3：用 model 做 $n$ 次 synthetic update**

從歷史 state-action 中隨機抽樣 $(s, a)$，用 model 產生虛擬 transition：

$$
\hat{s}' = f_\phi(s, a), \qquad \hat{r} = g_\phi(s, a)
$$

再用這筆虛擬資料更新 Q：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ \hat{r} + \gamma \max_{a'} Q(\hat{s}', a') - Q(s, a) \right]
$$

重複 $n$ 次。

### 為什麼這樣有用？

直覺很簡單：每一次真實互動只給你一筆資料，但如果 world model 夠準，你可以從那一次互動中**榨出 $n$ 筆額外的訓練訊號**。

更精確的說法是，Dyna-Q 相當於把 model-free 的 sample efficiency 提高了大約 $n$ 倍，前提是 model bias 不太嚴重。

### Dyna-Q 的收斂性

在 tabular setting、模型完全正確（$\hat{s}' = s'$, $\hat{r} = r$）的理想情況下，可以證明 Dyna-Q 的 Q function 會收斂到 $Q^*$。

原因是：model-based 的 synthetic update 和 model-free 的真實 update 在數學形式上是完全一樣的，差別只在資料來源。只要模型正確，兩者的固定點是同一個 $Q^*$。

$$
\text{若 } f_\phi = P(\cdot | s, a) \text{ 且 } g_\phi = R(s, a)，\text{則 Dyna-Q} \xrightarrow{t \to \infty} Q^*
$$

但若模型有誤差，Dyna-Q 的 Q function 會收斂到一個偏移後的固定點，而不是真正的 $Q^*$。這就是 model bias 的直接體現。

---

## Dreamer：在夢境裡訓練一切

Dreamer（Hafner et al., 2019/2020）把 latent world model 推到了一個極端：**整個 policy 和 value 的訓練，都在 latent 空間的 imagined rollout 裡完成，完全不接觸真實環境的梯度。**

### 架構

Dreamer 有三個核心模組：

$$
\text{Encoder: } z_t = e_\phi(o_t)
$$

$$
\text{Transition model: } z_{t+1} \sim p_\phi(z_{t+1} | z_t, a_t)
$$

$$
\text{Reward model: } \hat{r}_t = g_\phi(z_t, a_t), \qquad \text{Value model: } \hat{v}_t = v_\psi(z_t)
$$

注意 transition 是**機率性的**（RSSM，Recurrent State Space Model）：

$$
p_\phi(z_{t+1} | z_t, a_t) = \mathcal{N}\left(\mu_\phi(z_t, a_t),\, \sigma^2_\phi(z_t, a_t) \cdot I\right)
$$

這讓模型可以表示環境的隨機性，而不只是點預測。

### 訓練分兩個階段

**Phase 1：訓練 world model（用真實資料）**

World model 的訓練目標來自 ELBO，類似 VAE 的推導：

先把生成模型寫清楚。對一段長度 $T$ 的軌跡，latent dynamics 的聯合分佈可以寫成：

$$
p_\phi(o_{1:T}, z_{1:T} \mid a_{1:T}) = p(z_1) \prod_{t=1}^{T} p_\phi(z_t \mid z_{t-1}, a_{t-1})\, p_\psi(o_t \mid z_t)
$$

再用變分 posterior $q_\phi(z_t \mid o_t, z_{t-1}, a_{t-1})$ 近似真實 posterior，則對數似然下界可寫成：

$$
\mathcal{L}_{WM} = \mathbb{E}_{(o_{1:T}, a_{1:T}) \sim \mathcal{D}} \left[
\sum_{t=1}^{T} \left(
\underbrace{\ln p_\psi(o_t | z_t)}_{\text{reconstruction}}
- \underbrace{\beta \cdot KL[q_\phi(z_t | o_t, z_{t-1}, a_{t-1}) \,\|\, p_\phi(z_t | z_{t-1}, a_{t-1})]}_{\text{regularization}}
\right)
\right]
$$


這個式子的結構很像標準 VAE，只是 latent transition 不再是單純的 i.i.d. prior，而是帶有時間遞迴的 dynamics prior。換句話說，Dreamer 的 world model 不是只學「壓縮觀測」，而是同時學「壓縮」與「在 latent 裡做時間推演」。
其中 $q_\phi$ 是 posterior（有看到真實 $o_t$），$p_\phi$ 是 prior（只有 latent 轉移）。KL 項讓 latent 的 posterior 不要偏離 prior 太遠，確保 latent space 有良好結構。

**Phase 2：在 imagined rollout 裡訓練 policy 和 value**

從目前 latent state $z_t$ 出發，讓 policy 在 world model 裡往前滾 $H$ 步：

$$
a_\tau \sim \pi_\theta(a_\tau | z_\tau), \qquad z_{\tau+1} \sim p_\phi(z_{\tau+1} | z_\tau, a_\tau)
$$

定義 $\lambda$-return（類似 TD($\lambda$)）：

$$
V_\lambda(z_\tau) = (1 - \lambda) \sum_{n=1}^{H-\tau-1} \lambda^{n-1} V_n(z_\tau) + \lambda^{H-\tau-1} V_H(z_\tau)
$$

其中 $n$ 步 return 是：

$$
V_n(z_\tau) = \mathbb{E}\left[ \sum_{k=0}^{n-1} \gamma^k \hat{r}_{\tau+k} + \gamma^n \hat{v}_\psi(z_{\tau+n}) \right]
$$

Policy 的目標是最大化期望 $\lambda$-return，用梯度直接穿過 world model 往回傳（reparameterization trick）：

$$
\mathcal{L}_\pi = -\mathbb{E}_{\text{imagined}} \left[ V_\lambda(z_\tau) \right]
$$

Value network 的目標是擬合 $\lambda$-return：

$$
\mathcal{L}_v = \mathbb{E}_{\text{imagined}} \left[ \left( \hat{v}_\psi(z_\tau) - \text{sg}(V_\lambda(z_\tau)) \right)^2 \right]
$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient。

> **Insight：** Dreamer 的核心貢獻是把「在 imagined world 裡做 policy gradient」變成可微的操作。因為 world model 是可微分的神經網路，梯度可以從 $V_\lambda$ 一路穿過 $H$ 步的 latent transition 傳回 policy 參數。這比 model-free policy gradient 的 REINFORCE 有更低的方差，因為它走的是確定性梯度路徑，而不是 score function 估計。

### Dreamer 的訓練迴圈

```
while True:
    # 從真實環境收集少量資料
    τ_real = rollout(env, π_θ)
    D ← D ∪ τ_real

    # 用真實資料更新 world model
    update(world_model, D)

    # 在 imagined world 裡訓練 policy 和 value
    for _ in range(imagination_steps):
        z ~ D  # 從 buffer 取 latent seed
        imagined_τ = world_model.rollout(z, π_θ, H)
        update(π_θ, v_ψ, imagined_τ)
```

真實互動只用來訓練 world model；policy 的訓練幾乎全在 imagination 裡完成。這讓 Dreamer 的 sample efficiency 遠高於 model-free 方法。

---

## MuZero：把 Tree Search 也搬進 Latent Space

MuZero（Schrittwieser et al., 2020）走的是另一條路。它不做 imagined rollout 訓練 policy，而是用 latent world model 支援 Monte Carlo Tree Search（MCTS）。

### 不假設 Game Rules

AlphaGo 和 AlphaZero 都需要環境提供完整的 game dynamics（合法落子、輸贏判斷）。MuZero 把這個前提拿掉：它從 observation 自己學 dynamics，不需要外部告知規則。

### 三個函數

$$
\text{Representation: } h_t = e_\phi(o_{1:t}) \qquad \text{（把歷史壓成 latent）}
$$

$$
\text{Dynamics: } h_{t+1}, \hat{r}_t = g_\phi(h_t, a_t) \qquad \text{（latent 轉移 + reward 預測）}
$$

$$
\text{Prediction: } \hat{\pi}_t, \hat{v}_t = f_\phi(h_t) \qquad \text{（policy prior + value 估計）}
$$

### MCTS 的角色

每次要選動作時，從當前 latent $h_t$ 出發，用 $g_\phi$ 在 latent space 內展開一棵搜尋樹。樹上的每個節點代表一個 hypothetical latent state，每條邊代表一個 action，分數用 $\hat{r} + \gamma \hat{v}$ 來算。

MCTS 跑完後，從統計分佈 $\pi_{MCTS}$ 中選動作，同時把這個分佈作為 policy 的訓練目標。

### 訓練目標

MuZero 的損失函數把三個目標加在一起，對 $K$ 步展開做 unrolled 訓練：

$$
\mathcal{L}(\phi) = \sum_{k=0}^{K} \left[
\underbrace{\ell^r(u_{t+k},\, \hat{r}_t^k)}_{\text{reward loss}}
+ \underbrace{\ell^v(z_{t+k},\, \hat{v}_t^k)}_{\text{value loss}}
+ \underbrace{\ell^\pi(\pi_{t+k},\, \hat{\pi}_t^k)}_{\text{policy loss}}
\right]
$$

這裡的 unrolled training 可以理解成：先從 representation $h_t$ 出發，用 dynamics model 一步一步展開 latent trajectory，然後在每一個展開點上同時對 reward、value、policy 三件事做 supervision。若把第 $k$ 步的展開狀態記成 $h_t^k$，則實作上就是在最小化

$$
\sum_{k=0}^{K} \Bigl(\ell^r(r_{t+k}, \hat{r}_t^k) + \ell^v(z_{t+k}, \hat{v}_t^k) + \ell^\pi(\pi_{t+k}, \hat{\pi}_t^k)\Bigr)
$$

這樣的多頭損失。它的意義不是單點預測準不準，而是整條 latent rollout 的「計畫品質」是否足夠好，能支撐後續的 MCTS 搜尋。

其中 $u_{t+k}$ 是真實 reward，$z_{t+k}$ 是 bootstrapped value target，$\pi_{t+k}$ 是 MCTS 統計出的搜尋分佈。

Policy loss 是 cross-entropy：

$$
\ell^\pi(\pi, \hat{\pi}) = -\sum_a \pi(a) \log \hat{\pi}(a)
$$

Value loss 用 $n$ 步 bootstrap：

$$
z_t = \sum_{k=0}^{n-1} \gamma^k u_{t+k} + \gamma^n \hat{v}_{t+n}
$$

### Dreamer vs MuZero

| 面向 | Dreamer | MuZero |
|---|---|---|
| 學習目標 | Imagined rollout → policy gradient | MCTS 搜尋 → policy + value distillation |
| 動作空間 | 連續（適合 continuous control） | 離散（適合棋盤遊戲）|
| Planning 方式 | 用梯度穿過 latent rollout | 用 MCTS 在 latent tree 裡搜尋 |
| 真實環境互動 | 少量，主要用於訓練 world model | 少量，主要用於 self-play 收集資料 |
| 適合場景 | DMControl、Atari | Go、Chess、Atari |

兩者的共同點是：所有的「想像」都發生在 latent space，而不是 observation space。

---

## Planning 的兩種時機

Model-based RL 的 planning 可以發生在兩個不同的時間點，兩者目標不一樣。

### Background Planning（訓練時）

在訓練期間，用 world model 產生 synthetic data，再用這些資料來更新 Q / policy / value。Dyna-Q 的 synthetic Q-update 是最典型的例子。

目的是提升 **sample efficiency**：讓每一次和真實環境的互動，榨出更多學習訊號。

### Decision-time Planning（決策時）

在測試時，收到當前 observation 之後，在行動之前先用 world model 做一輪快速的前瞻搜尋，選出最佳 action，然後再真正執行。MuZero 的 MCTS 是最典型的例子。

目的是提升 **決策品質**：在不更新任何參數的情況下，用更多計算換取更好的 action。

這兩種 planning 並不互斥，很多系統同時使用。

---

## 動作空間對 Planning 的影響

### 離散動作：Tree Search 天然適配

如果 $|\mathcal{A}|$ 有限，可以在每個 hypothetical state 列舉所有下一步。這就是 MCTS 的基礎——從根節點出發，沿著有希望的分支展開，用 value estimate 截斷。

每個 tree node 維護 UCB score：

$$
\text{UCB}(s, a) = Q(s, a) + c \cdot \hat{\pi}(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}
$$

其中 $N(s)$ 是節點被訪問次數，$N(s, a)$ 是該動作被選次數，$c$ 是探索係數。這個公式平衡了利用（高 Q 值）和探索（訪問少的 action）。

### 連續動作：Optimization 是主路線

如果 $\mathcal{A} \subseteq \mathbb{R}^d$，枚舉就不可行。通常用以下幾種方式：

**梯度法（適用於可微模型）：**

$$
a^* = \arg\max_{a} \hat{Q}(s, a; \phi), \qquad \nabla_a \hat{Q}(s, a; \phi) \text{ 可算}
$$

**CEM（Cross-Entropy Method）：**

從 action 分佈抽 $N$ 個候選，保留前 $k$ 個（elite set），用它們更新分佈，反覆迭代：

$$
\mu_{i+1} = \frac{1}{k}\sum_{j \in \text{elite}} a_j, \qquad \sigma^2_{i+1} = \frac{1}{k}\sum_{j \in \text{elite}} (a_j - \mu_{i+1})^2
$$

CEM 不需要梯度，對非光滑的 world model 更穩定。

---

## 模型家族：Parametric vs Non-parametric

### Parametric Models

明確用參數 $\phi$ 描述的模型。

**System identification（物理系統）：** 如果已知系統是某種 ODE，例如雙擺：

$$
\ddot{\theta} = f(\theta, \dot{\theta}, a; \phi)
$$

只需要用資料 fit 未知的物理參數（質量、阻尼等）。這在 robotics 很常見，因為硬體廠商通常會提供 CAD 模型，只需要微調幾個參數。

**Neural dynamics model：** 用神經網路直接近似轉移函數，不假設任何物理結構，表達力更強，但需要更多資料。

**Gaussian Process Dynamics：** 把 $(s_t, a_t) \mapsto s_{t+1}$ 視為 GP regression：

$$
s_{t+1} \sim \mathcal{GP}(\mu_\phi(s_t, a_t),\, k_\phi(s_t, a_t;\, s_t, a_t))
$$

GP 的核心優點是**不確定性估計是自帶的**：在資料稀疏的區域，posterior variance 自然偏大，可以直接用來判斷是否該繼續 rollout 還是回到真實環境。缺點是計算複雜度是 $O(n^3)$，高維 state 難以 scale。

### Non-parametric Models

不用顯式參數表示模型，而是讓資料本身充當模型。

**Replay buffer 作為隱式 model：** replay buffer 儲存大量 $(s, a, r, s')$，其實隱含了環境動態的 empirical distribution。在某些設計下，直接從 buffer 裡 sample past transition 來做 planning，等於在用一個 non-parametric 的轉移模型。

**語言 / 符號系統：** PDDL 用 predicate logic 描述環境：pre-conditions、effects、goal。這是一種高度結構化的 world model，planning 就是在邏輯空間做搜尋。近年來也有工作用 LLM 作為 world model，直接問 "if I do $a$ in state $s$, what happens?"。

---

## 完整 Loss 設計

以 Dreamer 架構為例，完整的 loss 函數結構是：

$$
\mathcal{L}_\text{total} = \underbrace{\mathcal{L}_{rec}}_{\text{重建 observation}} + \underbrace{\mathcal{L}_{kl}}_{\text{latent 正則化}} + \underbrace{\mathcal{L}_{rew}}_{\text{reward 預測}} + \underbrace{\mathcal{L}_{v}}_{\text{value 學習}} + \underbrace{\mathcal{L}_\pi}_{\text{policy 更新}}
$$

展開各項：

$$
\mathcal{L}_{rec} = \mathbb{E}\left[\sum_t -\ln p_\psi(o_t | z_t)\right]
$$

$$
\mathcal{L}_{kl} = \beta \cdot \mathbb{E}\left[\sum_t KL\left[q_\phi(z_t | o_t, z_{t-1}) \,\|\, p_\phi(z_t | z_{t-1}, a_{t-1})\right]\right]
$$

$$
\mathcal{L}_{rew} = \mathbb{E}_\text{imagined}\left[\sum_\tau (\hat{r}_\tau - r_\tau)^2\right]
$$

$$
\mathcal{L}_v = \mathbb{E}_\text{imagined}\left[\sum_\tau \left(\hat{v}_\psi(z_\tau) - V_\lambda(z_\tau)\right)^2\right]
$$

$$
\mathcal{L}_\pi = -\mathbb{E}_\text{imagined}\left[\sum_\tau V_\lambda(z_\tau)\right]
$$

注意 $\mathcal{L}_{rec}$ 和 $\mathcal{L}_{kl}$ 只用**真實資料**訓練，$\mathcal{L}_{rew}$、$\mathcal{L}_v$、$\mathcal{L}_\pi$ 在 **imagined rollout** 上計算。

---

## 常見誤解

**誤解 1：Model-based 一定比 Model-free 樣本效率高。**

不一定。如果 world model 需要大量資料才能學準，那前期的 model bias 可能讓 agent 在錯誤的方向上訓練很久。事實上，在高度隨機或部分可觀測的環境，model-free 有時反而更穩定。

**誤解 2：World model 越複雜越好。**

不對。世界模型只需要捕捉 task-relevant 的資訊。把背景像素都重建得一清二楚，對決策沒幫助，只是浪費計算資源。Latent model 刻意放棄細節，正是這個道理。

**誤解 3：Dreamer 是「不和真實環境互動」的算法。**

不是。Dreamer 仍然和真實環境互動，只是互動頻率很低。真實資料用來訓練 world model，policy 的訓練才是靠 imagination。

**誤解 4：Model error 可以用更多 synthetic data 彌補。**

不行。Synthetic data 的品質上限是 world model 本身的準確度。如果 model 有系統性偏差，跑再多的 imagined rollout 也只是在放大這個偏差。真正的解法是回到真實環境蒐集更多資料，修正 world model。

---

## 總結

1. World model 的本質是讓 agent 能在「腦中」模擬環境，從而做前瞻性規劃，而不是單純靠反應式的 value/policy 決策。

2. Compound error 是 model-based RL 的核心難點：單步誤差 $\epsilon$ 在 $H$ 步 rollout 後可能累積成 $O(\epsilon L^H)$ 的偏差，因此 rollout 長度、model 不確定性估計、和真實環境的互動頻率都需要仔細設計。

3. Dyna-Q 用最簡單的方式驗證了核心想法：一次真實互動 $+$ $n$ 次 synthetic update，在 model 準確的前提下可以大幅提升 sample efficiency。

4. Dreamer 把 latent world model 和可微分 policy gradient 結合：world model 用 ELBO 訓練，policy 和 value 靠 imagined $\lambda$-return 更新，梯度可以穿過 latent rollout 直接傳回去。

5. MuZero 走另一條路：latent dynamics 支援 MCTS，搜尋結果反過來訓練 policy prior 和 value，適合離散動作的精確規劃任務。

6. 選擇 model-based 的正確時機是：環境互動成本高、任務需要長期規劃、或環境有明顯可利用的結構。如果這三點都不成立，model-free 的穩定性可能是更務實的選擇。