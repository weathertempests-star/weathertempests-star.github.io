---
type: Blog-post
tags:
  - Deep-Learning
  - note
  - Reinforcement-Learning
created: 2026-05-28
updated: 2026-05-29
lastmod: 2026-06-01
authors: WeiTa
status: active
---

---


## 這篇文章想回答什麼問題？

讀完 value-based RL 之後，一個很自然的問題是：

> 我們已經知道怎麼學 $Q^*$ 了，為什麼還需要另一套方法？

這篇文章想回答的，就是這個問題。更具體地說，是四個環環相扣的問題：

1. Value-based 方法在什麼情況下會遇到根本性的困難？
2. 把 RL 寫成直接對策略參數求梯度的問題，數學上要怎麼做到？
3. Policy gradient 的 variance 為什麼很大，Baseline 和 Advantage 怎麼拯救它？
4. 從 REINFORCE 到 Actor-Critic、PPO、RLHF，這條演化鏈的邏輯是什麼？

如果把整篇濃縮成一句話：

> Policy-based RL 不是 value-based 的替代品，而是在另一個層次上直接把「決策方式」本身當成可訓練的對象；它的核心技巧是 log-derivative trick，而它的核心挑戰是如何在高 variance 下仍然學得穩。

---

## 為什麼 Value-based 不夠用？

### 問題一：連續動作空間讓 argmax 無解

Value-based 的策略改善步驟是：

$$
\pi_{k+1}(s) = \arg\max_{a \in \mathcal{A}} Q^{\pi_k}(s, a)
$$

這個 argmax 的前提是：動作空間 $\mathcal{A}$ 可以枚舉。

但現實中很多控制任務的動作是連續的。機械手臂的關節角度、自駕車的方向盤轉角、無人機的三軸力矩——這些都是 $\mathbb{R}^n$ 上的連續向量。在連續空間裡做 argmax，本身就是一個非凸最佳化問題，不是掃一遍就能解決的。

### 問題二：Greedy improvement 的隱含假設

即使動作是離散的，DQN 的做法是：

$$
a^* = \arg\max_{a} \hat{Q}(s, a; w)
$$

這代表每次選動作時都需要把所有動作都評估一遍，再取最大。當動作空間很大（例如影像生成、自然語言生成），這顯然不可行。

### 問題三：確定性策略的表達受限

還有一類問題，最佳策略本身就是隨機的（stochastic）。例如石頭剪刀布的最佳策略，就是三者各取 $1/3$ 的混合策略；在 POMDP 裡，隨機策略通常也比確定性策略更優。Value-based 的 greedy policy 是確定性的，天然不適合這類任務。

> **這三個問題指向同一個修補方向**：與其先學 Q function 再反推策略，不如直接把策略本身參數化，讓「怎麼做」變成學習的對象。

---

## 核心想法：把策略本身變成目標函數的對象

把策略參數化為 $\pi_\theta$，然後把 RL 的目標直接寫成：

$$
\max_\theta \; V^{\pi_\theta}(\mu) = \mathbb{E}_{\tau \sim P_\mu^{\pi_\theta}}[G(\tau)]
$$

其中 $\mu$ 是初始狀態分佈，$G(\tau) = \sum_{t=0}^{\infty} \gamma^t r_{t+1}$ 是整條軌跡的折扣回報。

這個寫法的重點是：我們先用策略 $\pi_\theta$ 與環境互動，得到一條軌跡 $\tau$，再對軌跡的總回報取期望。也就是說，優化目標不是單一步的 reward，而是整個策略誘導出的長期行為品質。

這個目標函數的直覺非常清楚：

- 如果一條軌跡最後回報高，就讓產生這條軌跡的動作更常出現。
- 如果回報低，就壓低那些動作的機率。

但問題在於：**這個目標函數的梯度，要怎麼算？**

軌跡 $\tau$ 是從環境採樣出來的，包含了不可微的環境動態 $P(s_{t+1} \mid s_t, a_t)$。我們沒辦法對「環境怎麼反應」求梯度。

解決這個問題的關鍵，是 **log-derivative trick**，也叫 likelihood ratio trick。

---

## 數學推導

### 設定

一條軌跡的機率分佈是：

$$
P_\mu^{\pi_\theta}(\tau) = \mu(s_0) \prod_{t=0}^{\infty} \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)
$$

其中 $\mu(s_0)$ 是初始狀態分佈，$P(s_{t+1} \mid s_t, a_t)$ 是環境轉移動態。

**關鍵觀察**：環境動態 $P$ 不含 $\theta$，所以對 $\theta$ 求梯度時，它不貢獻任何梯度。

---

### Step 1：Log-Derivative Trick

對目標函數直接求梯度：

$$
\nabla_\theta V^{\pi_\theta}(\mu) = \nabla_\theta \sum_\tau P_\mu^{\pi_\theta}(\tau) \cdot G(\tau)
= \sum_\tau G(\tau) \cdot \nabla_\theta P_\mu^{\pi_\theta}(\tau)
$$

這裡有一個恆等式：

$$
\nabla_\theta P = P \cdot \nabla_\theta \log P
$$

把它代入：

$$
\nabla_\theta V^{\pi_\theta}(\mu) = \sum_\tau P_\mu^{\pi_\theta}(\tau) \cdot G(\tau) \cdot \nabla_\theta \log P_\mu^{\pi_\theta}(\tau)
= \mathbb{E}_{\tau \sim P_\mu^{\pi_\theta}} \left[ G(\tau) \cdot \nabla_\theta \log P_\mu^{\pi_\theta}(\tau) \right]
$$

這個移動的好處是：**原本需要對 $P$ 求梯度，現在只需要對 $\log P$ 求梯度，而且包在期望值裡可以用 Monte Carlo 樣本估計。**

---

### Step 2：展開 log 軌跡機率

$$
\log P_\mu^{\pi_\theta}(\tau) = \log \mu(s_0) + \sum_{t=0}^{\infty} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{\infty} \log P(s_{t+1} \mid s_t, a_t)
$$

對 $\theta$ 求梯度，第一項（$\mu$）和第三項（環境轉移 $P$）都不含 $\theta$，梯度為零。剩下：

$$
\nabla_\theta \log P_\mu^{\pi_\theta}(\tau) = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

代回得到 **Policy Gradient Theorem（P1 版本，總回報）**：

$$
\boxed{
\nabla_\theta V^{\pi_\theta}(\mu) = \mathbb{E}_{\tau \sim P_\mu^{\pi_\theta}} \left[ G(\tau) \cdot \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
}
$$

這就是最原始的 policy gradient。它說的是：整條軌跡如果最後很成功，就把這條軌跡上所有動作的機率一起推高；如果很失敗，就一起壓低。

---

### Step 3：Causality 論點 → REINFORCE

P1 版本有個明顯缺點：同一條軌跡上，時間步 $t=0$ 的動作和 $t=100$ 的動作，都拿到同一個全局 return $G(\tau)$，這顯然不合理——時間步 $t$ 的動作不應該對它之前的 reward 負責。

用條件期望的語言來說：

$$
\mathbb{E}_\tau \left[ r_k \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = 0, \quad \text{for } k < t
$$

因為動作 $a_t$ 和 $k < t$ 之前的回報是條件獨立的（因果性）。

所以可以把 $G(\tau)$ 換成 **return-to-go** $G_t = \sum_{k=t}^{\infty} \gamma^{k-t} r_{k+1}$，得到 **REINFORCE / P2 版本**：

$$
\boxed{
\nabla_\theta V^{\pi_\theta}(\mu) = \mathbb{E}_{\tau \sim P_\mu^{\pi_\theta}} \left[ \sum_{t=0}^{\infty} G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
}
$$

再把 $G_t$ 用動作價值函數表示（$Q^{\pi_\theta}(s_t, a_t) = \mathbb{E}[G_t \mid s_t, a_t]$），加上折扣因子 $\gamma^t$：

$$
\nabla_\theta V^{\pi_\theta}(\mu) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{\infty} \gamma^t Q^{\pi_\theta}(s_t, a_t) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

---

### Step 4：分佈視角 → Policy Gradient Theorem 正式版（P3）

把對時間的求和，改寫成對 discounted state visitation distribution 的期望。

定義折扣狀態訪問頻率：

$$
d_\mu^{\pi_\theta}(s) = \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s \mid \mu, \pi_\theta)
$$

它代表在策略 $\pi_\theta$ 下，長期互動中 state $s$ 被以折扣加權訪問的頻率。

用 $d_\mu^{\pi_\theta}$ 可以把對時間的求和重寫成對 state 的積分，得到 **P3 版本**：

$$
\boxed{
\nabla_\theta V^{\pi_\theta}(\mu) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d_\mu^{\pi_\theta}, \, a \sim \pi_\theta(\cdot \mid s)} \left[ Q^{\pi_\theta}(s, a) \cdot \nabla_\theta \log \pi_\theta(a \mid s) \right]
}
$$

這裡的 $1/(1-\gamma)$ 是把折扣時間的總權重正規化後得到的常數；而 $d_\mu^{\pi_\theta}(s)$ 則把「沿著時間往前加總」改寫成「對狀態訪問分佈取期望」。因此 P3 不是另一個全新的梯度，而是 P2 在分佈視角下的等價表達。

> 這三個版本不是三個不同的定理，而是同一件事在「時間域」與「分佈域」的不同觀點。實作時通常用 P2（REINFORCE）或加上 baseline 的變形。

---

## Variance 的問題與 Baseline

Policy gradient 的理論很漂亮，但實作起來通常不穩定。問題出在 **variance 極大**。

### 為什麼 variance 大？

Monte Carlo 估計 $G_t$ 時，同樣的策略跑出來的軌跡可能差異極大——某次運氣好，回報很高；某次運氣差，回報很低。這個隨機性直接乘進梯度估計，讓每次更新的方向和大小都不一樣。

### Baseline：零偏差的免費降噪

一個關鍵觀察是：可以在梯度估計裡加一個 baseline $B(s_t)$ 而**不改變梯度的期望值**：

$$
\mathbb{E}_\tau \left[ B(s_t) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = 0
$$

**完整證明**：

$$
\begin{aligned}
\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)} \left[ B(s) \cdot \nabla_\theta \log \pi_\theta(a \mid s) \right]
&= B(s) \sum_a \pi_\theta(a \mid s) \cdot \nabla_\theta \log \pi_\theta(a \mid s) \\
&= B(s) \sum_a \pi_\theta(a \mid s) \cdot \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} \\
&= B(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) \\
&= B(s) \cdot \nabla_\theta \underbrace{\sum_a \pi_\theta(a \mid s)}_{=1} \\
&= 0
\end{aligned}
$$

所以加了 baseline 的梯度估計式：

$$
\nabla_\theta V^{\pi_\theta} = \mathbb{E}_\tau \left[ \sum_{t} \left( G_t - B(s_t) \right) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

仍然是無偏估計，但 variance 可以大幅降低。

### 最佳 Baseline 是 Value Function

從變異數最小化的角度推導，最佳 baseline 是 state value function：

$$
B^*(s) = V^{\pi_\theta}(s) = \mathbb{E}_{a \sim \pi_\theta}[Q^{\pi_\theta}(s, a)]
$$

這讓我們把 $(G_t - B(s_t))$ 改寫成 **Advantage function**：

$$
A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)
$$

代入後，policy gradient 的 advantage form：

$$
\boxed{
\nabla_\theta V^{\pi_\theta}(\mu) = \frac{1}{1-\gamma} \mathbb{E}_{s, a} \left[ A^{\pi_\theta}(s, a) \cdot \nabla_\theta \log \pi_\theta(a \mid s) \right]
}
$$

**Advantage 的語意**：它回答的不是「這個動作本身好不好」，而是「這個動作比在這個 state 下的平均水準好多少」。

- $A > 0$：這個動作比平均好，應該更常選。
- $A < 0$：這個動作比平均差，應該少選。
- $A = 0$：這個動作和平均一樣，不需要推動任何方向。

---

## Deterministic Policy Gradient（DPG）

如果策略不是隨機分佈，而是直接輸出一個確定的動作：

$$
a_t = \pi_\theta(s_t)
$$

那麼 policy gradient 的推導就不能再對 action 做積分了。

**DPG 定理**告訴我們：

$$
\nabla_\theta V^{\pi_\theta}(\mu) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d_\mu^{\pi_\theta}} \left[ \underbrace{\nabla_\theta \pi_\theta(s)}_{\text{policy Jacobian}} \cdot \underbrace{\nabla_a Q^{\pi_\theta}(s, a) \big|_{a = \pi_\theta(s)}}_{\text{Q 對 action 的梯度}} \right]
$$

幾何解讀：

- $\nabla_a Q^{\pi_\theta}(s, a)$：在當前 state 下，動作往哪個方向調整，Q 值會上升。
- $\nabla_\theta \pi_\theta(s)$：策略參數怎麼改，才能讓輸出動作往那個方向移動。

DPG 本質上是透過 Q function 的梯度，把「動作空間的方向」映射回「參數空間的更新方向」。這讓它在連續動作任務上特別有效，DDPG、TD3 都是這個想法的延伸。

---

## 從 REINFORCE 到 Actor-Critic

### 純 Monte-Carlo：REINFORCE

最簡單的做法：跑完整條 episode，用真實回報 $G_t$ 作為估計：

$$
\nabla_\theta V \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t} G_t^{(n)} \cdot \nabla_\theta \log \pi_\theta(a_t^{(n)} \mid s_t^{(n)})
$$

**優點**：無偏估計。

**缺點**：

- 必須等整條 episode 結束。
- $G_t$ 的 variance 隨序列長度指數成長，訓練極不穩定。

### 單步 TD：用 Critic 替換 Monte-Carlo

引入 value network（critic）估計 $V^{\pi_\theta}(s)$，把 $G_t$ 替換成 one-step TD target：

$$
\hat{A}_t^{\text{TD}(0)} = \underbrace{r_{t+1} + \gamma V_w(s_{t+1})}_{\text{TD target}} - V_w(s_t)
$$

這就是 **TD(0) advantage**，也叫 one-step advantage。

**優點**：可以線上更新，variance 低。

**缺點**：因為用 bootstrapping（以估計值估計估計值），引入了 bias。

### Generalized Advantage Estimation（GAE）

GAE 是 MC 和 TD 之間的連續插值，用來平衡 bias 和 variance。

先定義 $n$ 步 TD error 的遞推：

$$
\delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)
$$

GAE 把不同步長的 advantage 做指數加權平均：

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\lambda \in [0, 1]$ 是插值係數：

| $\lambda$ | 行為 | 特性 |
|-----------|------|------|
| $\lambda = 0$ | 退化成 one-step TD | 低 variance，高 bias |
| $\lambda = 1$ | 退化成 Monte-Carlo | 無 bias，高 variance |
| $0 < \lambda < 1$ | 中間插值 | 可控 bias-variance trade-off |

> **推導 $\lambda = 1$ 退化成 MC 的細節**：把 $\delta_t$ 展開後相加，相鄰項的 $\gamma V_w(s_{t+1})$ 和 $-V_w(s_{t+1})$ 會消掉（telescoping sum），最後剩下 $G_t - V_w(s_t)$，正是 MC advantage。

若把有限長 episode 的上限寫成 $T$，更完整地展開會得到：

$$
\begin{aligned}
\sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l}
&= \sum_{l=0}^{T-t-1} \gamma^l \bigl(r_{t+l+1} + \gamma V_w(s_{t+l+1}) - V_w(s_{t+l})\bigr) \\
&= \sum_{l=0}^{T-t-1} \gamma^l r_{t+l+1} - V_w(s_t) + \gamma^{T-t} V_w(s_T)
\end{aligned}
$$

若終止狀態滿足 $V_w(s_T)=0$，就立刻回到 $G_t - V_w(s_t)$。這就是 GAE 在 $\lambda=1$ 時與 Monte-Carlo advantage 相接的原因。

實作時通常截斷到有限步 $T$，並在實際訓練中 $\lambda \in [0.9, 0.99]$。

### Actor-Critic（A2C）

A2C 把 policy gradient 和 value estimation 整合成同一個架構：

- **Actor**：$\pi_\theta(a \mid s)$，策略網路，負責「做什麼」。
- **Critic**：$V_w(s)$，價值網路，負責「做得好不好」。

更新規則：

$$
\theta \leftarrow \theta + \alpha_\theta \sum_t \hat{A}_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

$$
w \leftarrow w - \alpha_w \sum_t \left( r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t) \right)^2
$$

Actor 對 advantage 做梯度上升；Critic 對 TD error 的平方做梯度下降。

---

## PPO：近端策略優化

### 問題：更新步伐太大會搞壞策略

Policy gradient 的梯度更新有一個潛在問題：如果 learning rate 太大，一次更新可能把策略推到一個很差的地方，接下來收集的資料又基於這個很差的策略，形成惡性循環。

一個量化「策略偏移」的方法是看 importance sampling ratio：

$$
r_\theta(s, a) = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}
$$

其中 $\theta_k$ 是更新前的舊策略參數。

理想中，我們想在策略改變不超過某個範圍的前提下，盡量提升 advantage：

$$
\max_\theta \; \mathbb{E}_{s, a \sim \pi_{\theta_k}} \left[ r_\theta(s, a) \cdot A^{\pi_{\theta_k}}(s, a) \right]
$$

### PPO-KL

加入 KL penalty，懲罰新舊策略之間的距離：

$$
\max_\theta \; \mathbb{E} \left[ r_\theta(s, a) \cdot A^{\theta_k}(s, a) - \beta \, D_{KL}\left(\pi_{\theta_k}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s)\right) \right]
$$

$\beta$ 可以自適應調整：如果 KL 太大就增大 $\beta$，如果太小就縮小 $\beta$。

### PPO-Clip（更常用）

PPO-Clip 不用 KL，而是直接把 importance ratio 裁剪在 $[1-\epsilon, 1+\epsilon]$ 的範圍內：

$$
\boxed{
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min \left( r_\theta \cdot A^{\theta_k}, \;\; \text{clip}(r_\theta, 1-\epsilon, 1+\epsilon) \cdot A^{\theta_k} \right) \right]
}
$$

這個目標函數的行為分兩種情況討論：

**當 $A > 0$（這個動作比平均好，想增加機率）**：

- $r_\theta > 1$：說明新策略比舊策略更常選這個動作。clip 上界是 $1+\epsilon$，超過這個就停止再增加。
- $r_\theta < 1$：說明新策略選這個動作的機率下降了，沒有 clip，讓它自由提升。

**當 $A < 0$（這個動作比平均差，想降低機率）**：

- $r_\theta < 1$：說明新策略比舊策略更少選這個動作。clip 下界是 $1-\epsilon$，低過這個就停止再降。
- $r_\theta > 1$：說明新策略選了更多這個不好的動作，沒有 clip，讓 loss 繼續懲罰。

取 min 的語意是：**不管哪種情況，都不讓更新的幅度超過 $\epsilon$ 的範圍**，確保「小步安全更新」的原則。

---

## 策略怎麼參數化？

策略的參數化方式決定了演算法能處理什麼類型的問題。

### 離散動作：Softmax

最直接的做法是讓網路輸出每個動作的 logit，再取 softmax：

$$
\pi_\theta(a \mid s) = \frac{\exp(f_\theta(s, a))}{\sum_{a'} \exp(f_\theta(s, a'))}
$$

此時 $\nabla_\theta \log \pi_\theta(a \mid s)$ 就是 softmax 的 gradient，有標準的解析形式。

### 連續動作：Gaussian Policy

最常見的連續策略是輸出動作分佈的 mean 和 variance：

$$
\pi_\theta(a \mid s) = \mathcal{N}\left(a;\, \mu_\theta(s),\, \sigma_\theta^2(s) \cdot I\right)
$$

神經網路輸出 $\mu_\theta(s)$（mean）和 $\log \sigma_\theta^2(s)$（log variance，確保正值）。

此時 $\nabla_\theta \log \pi_\theta(a \mid s)$ 可以直接計算：

$$
\log \pi_\theta(a \mid s) = -\frac{(a - \mu_\theta(s))^2}{2\sigma_\theta^2(s)} - \frac{1}{2}\log(2\pi\sigma_\theta^2(s))
$$

$$
\nabla_\theta \log \pi_\theta(a \mid s) = \frac{(a - \mu_\theta(s))}{\sigma_\theta^2(s)} \cdot \nabla_\theta \mu_\theta(s) + \text{（variance 項的梯度）}
$$

### Reparameterization

另一種做法：不直接從 $\pi_\theta(a \mid s)$ 採樣，而是：

$$
a = f_\theta(s, \epsilon), \quad \epsilon \sim p(\epsilon)
$$

例如對 Gaussian policy：$a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

這樣 $a$ 本身是 $\theta$ 的確定性函數，可以直接對 $\theta$ 做 backpropagation，不需要估計 policy gradient。這是 SAC（Soft Actor-Critic）等算法的核心技巧。

---

## RLHF：用人類偏好學 Reward

當任務的 reward 難以手動定義（例如「生成好的文章」），可以先收集人類偏好樣本，訓練一個 reward model。

### Reward Model 的訓練

給定 prompt $x$，和兩個回應 $y_w$（preferred）、$y_l$（less preferred），用 Bradley-Terry model 描述人類偏好的機率：

$$
P(y_w \succ y_l \mid x) = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

對應的訓練損失（最大化 log-likelihood）：

$$
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]
$$

### 用 PPO 微調語言模型

Reward model 訓練好之後，把它接回 PPO，把語言模型當成 policy：

$$
\max_\theta \; \mathbb{E}_{x, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \, D_{KL}\left(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\right) \right]
$$

其中 $\pi_{\text{ref}}$ 是微調前的參考模型（例如 SFT 模型），KL penalty 防止模型偏離太遠。

> Insight：RLHF 從架構上來說，就是 policy gradient + learned reward。語言模型的每一個生成 token 對應一個「動作」，整個回應生成過程對應一條「軌跡」，人類的偏好評分對應「延遲稀疏 reward」。

---

## 常見誤解

**誤解一：Baseline 會讓梯度有偏差。**

不會。我們已經完整證明 $\mathbb{E}[B(s_t) \cdot \nabla_\theta \log \pi_\theta] = 0$，任何只依賴 state 的 baseline 都是零偏差的，只影響 variance。

**誤解二：Policy gradient 就是 REINFORCE，這兩個名字可以互換。**

不完全對。Policy gradient theorem 是一個關於梯度結構的數學結論；REINFORCE 是它的一種具體估計方法（用 Monte-Carlo 回報代替 $Q$）。Actor-Critic、PPO 都是 policy gradient theorem 的後代，但不叫 REINFORCE。

**誤解三：PPO 只是把 policy gradient 加了個 clip，本質沒有差別。**

差別很大。Clip 機制確保了每次更新的策略偏移量有上界，讓訓練過程穩定不崩。這件事在大規模系統（例如訓練 LLM）裡至關重要。

**誤解四：Deterministic policy gradient 需要對環境求梯度。**

不需要。DPG 是透過 $\nabla_a Q(s, a)$ 來判斷動作應該往哪個方向調整，而 Q function 是用 TD 方法學出來的，不涉及環境動態的梯度。

---

## 演化鏈整理

```
Value-based RL（Q-learning / DQN）
    └──> 連續動作 / 大動作空間 → 失敗

Policy Gradient Theorem（P1 總回報）
    └──> Causality 剪枝 → REINFORCE（P2 return-to-go）
    └──> 分佈視角 → P3（discounted state visitation）

REINFORCE
    └──> 高 variance → 加 Baseline / Advantage

Advantage + Monte-Carlo Critic
    └──> 必須等 episode 結束 → 引入 TD(0) 估計

TD(0) advantage
    └──> bias-variance 難以控制 → GAE（$\lambda$ 插值）

GAE + Actor-Critic（A2C）
    └──> 更新步伐過大，策略不穩定 → PPO-Clip

PPO
    └──> reward 難以設計 → RLHF（learned reward + PPO）
```

---

## 總結

1. **Policy-based RL 的動機**：連續動作空間讓 argmax 無解，大動作空間讓窮舉不可行，部分任務的最佳策略本身就是隨機的。這三個問題讓「先學 Q、再反推策略」的路線受限。

2. **Policy gradient 的核心**：Log-derivative trick 把「對軌跡機率求梯度」轉化成「對 $\log \pi_\theta$ 求梯度後放進期望值」，讓 Monte Carlo 估計成為可能。Causality 論點把 $G(\tau)$ 換成 return-to-go $G_t$，消去未來動作對過去 reward 的虛假依賴。

3. **Baseline 和 Advantage**：任何只依賴 state 的 baseline 不改變梯度期望，但可以大幅降低 variance。Advantage $A(s, a) = Q(s, a) - V(s)$ 是最自然的 baseline，回答的是「這個動作比平均水準好多少」。

4. **GAE**：$\lambda \in [0, 1]$ 在 MC 和 TD 之間做插值，$\lambda = 0$ 是純 TD（低 variance、高 bias），$\lambda = 1$ 是純 MC（無 bias、高 variance），實作中通常取 $\lambda \approx 0.95$。

5. **PPO-Clip**：用 importance ratio $r_\theta = \pi_\theta / \pi_{\theta_k}$ 和 clip 機制，限制每次更新的策略偏移量，讓 policy improvement 變成「小步且安全」的迭代過程。

6. **RLHF**：把人類偏好轉成 reward model，再用 PPO 微調策略，讓語言模型的生成品質可以對齊人類判斷。這件事的底層仍然是 policy gradient。