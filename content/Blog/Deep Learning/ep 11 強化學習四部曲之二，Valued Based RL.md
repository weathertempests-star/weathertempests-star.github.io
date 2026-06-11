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


## 這篇文章想回答什麼問題？


上一篇文章把 RL 的基本架構交代清楚了：MDP 給了問題的數學語言，Bellman 方程把無限長的 return 壓成了遞迴，Policy Iteration 和 Value Iteration 則在「知道環境模型」的前提下找到最優策略。

但真實世界裡，你幾乎不可能知道完整的 $P(s' \mid s, a)$ 和 $R(s, a)$。你能拿到的，只有和環境互動之後的一筆一筆軌跡：

$$
(s_t,\; a_t,\; r_{t+1},\; s_{t+1})
$$

Value-based RL 要解的問題，就是：

> 只靠這種樣本，能不能把 $Q^*(s, a)$ 學出來？然後再由它還原出最優策略？

這篇文章會圍繞四條主線展開：

1. 為什麼能從樣本逼近 Bellman 方程？Q-learning 的數學根基是什麼？
2. Q-learning 什麼情況下能收斂，收斂條件從哪裡來？
3. 函數近似一旦接上神經網路，為什麼訓練會不穩？DQN 的兩個裝置在修什麼？
4. 離散動作空間之外，連續動作該怎麼辦？

如果用一句話濃縮：

> Value-based RL 的核心是用 **Bellman error** 當作學習信號，從互動樣本出發，逐步把 $Q^*$ 的估計修準，最後 greedy 出最優策略。

---

## 從 Bellman 方程到學習演算法

### 回顧：$Q^*$ 滿足什麼方程式

從上一篇的推導，最優 Q 函數滿足 Bellman optimality equation：

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')
$$

定義 Bellman optimality operator $\mathcal{T}^*$：

$$
(\mathcal{T}^* Q)(s, a) := R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')
$$

則 $Q^*$ 是 $\mathcal{T}^*$ 的唯一不動點：$\mathcal{T}^* Q^* = Q^*$。

**為什麼可以迭代求解？** 因為 $\mathcal{T}^*$ 是 $\ell^\infty$ 範數下的 $\gamma$-收縮映射（$\gamma < 1$）：

$$
\|\mathcal{T}^* Q - \mathcal{T}^* Q'\|_\infty \le \gamma \|Q - Q'\|_\infty
$$

**證明：**

$$
\begin{aligned}
|(\mathcal{T}^* Q)(s,a) - (\mathcal{T}^* Q')(s,a)|
&= \gamma \left| \sum_{s'} P(s'|s,a) \Bigl(\max_{a''} Q(s',a'') - \max_{a''} Q'(s',a'') \Bigr) \right| \\
&\le \gamma \sum_{s'} P(s'|s,a) \left| \max_{a''} Q(s',a'') - \max_{a''} Q'(s',a'') \right| \\
&\le \gamma \sum_{s'} P(s'|s,a) \max_{a''} |Q(s',a'') - Q'(s',a'')| \\
&\le \gamma \|Q - Q'\|_\infty
\end{aligned}
$$

其中第三個不等式用了 $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$。

由 Banach 不動點定理，從任意初始 $Q_0$ 出發，反覆套用 $\mathcal{T}^*$：

$$
Q_{k+1} = \mathcal{T}^* Q_k
$$

都會以幾何速率收斂到 $Q^*$，誤差每步縮小至少 $\gamma$ 倍：

$$
\|Q_k - Q^*\|_\infty \le \gamma^k \|Q_0 - Q^*\|_\infty
$$

這是 Q-value Iteration 在已知模型時的理論保證。

### 沒有模型時：用樣本近似期望

Value Iteration 的問題是每次更新都要對所有 $s'$ 求加權和，這需要知道 $P(s'|s,a)$。

現在換一個視角。把期望拆開：

$$
Q^*(s,a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}\!\left[R(s,a) + \gamma \max_{a'} Q^*(s', a')\right]
$$

如果我們真的去環境裡執行了動作 $a_t$ 並觀察到 $(r_{t+1}, s_{t+1})$，那麼：

$$
y_t := r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')
$$

就是一個對 $(\mathcal{T}^* Q)(s_t, a_t)$ 的**無偏估計**（假設 $Q$ 固定）。

於是「更新 $Q$ 讓它更接近 $\mathcal{T}^* Q$」這件事，就可以用單一樣本取代期望：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \underbrace{\bigl[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\bigr]}_{\text{TD error}\;\delta_t}
$$

這就是 **Q-learning**。

---

## Q-learning 的數學剖析

### TD Error 是什麼？

TD error（Temporal Difference error）$\delta_t$ 的結構很清楚：

$$
\delta_t = \underbrace{r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{target（觀察新資訊後的估計）}} - \underbrace{Q(s_t, a_t)}_{\text{舊估計}}
$$

- $\delta_t > 0$：實際比預期好，把 $Q(s_t, a_t)$ 往上修。
- $\delta_t < 0$：實際比預期差，往下修。
- $\delta_t = 0$：當前估計恰好符合 Bellman 方程，不動。

每次更新，$Q(s_t, a_t)$ 往 target 方向走 $\alpha$ 步。就像跟著風向微調羅盤——一次的樣本可能有雜訊，但方向平均起來是對的。

### 損失函數的視角

把 Q-learning 改寫成最小化問題，定義 Bellman error 的平方損失：

$$
\mathcal{L}(w) = \mathbb{E}_{(s,a,r,s') \sim \rho}\!\left[\frac{1}{2}\bigl(r + \gamma \max_{a'} Q(s',a';w) - Q(s,a;w)\bigr)^2\right]
$$

對 $Q(s,a;w)$ 取偏導（**注意：target 那一項視為常數，不對 $w$ 求導**）：

$$
\nabla_w \mathcal{L} = -\mathbb{E}\!\left[\delta_t \cdot \nabla_w Q(s_t,a_t;w)\right]
$$

這個式子可以直接讀成：如果把 bootstrap target 視為暫時固定，那麼每一步更新就是在最小化單一樣本的平方誤差 $\frac{1}{2}\delta_t^2$。也就是說，Q-learning 並不是在對整個 Bellman 方程做「一次解完」，而是在樣本層級反覆做小步修正，讓目前的估計逐漸逼近不動點。

沿著負梯度方向更新：

$$
w \leftarrow w + \alpha \cdot \delta_t \cdot \nabla_w Q(s_t,a_t;w)
$$

在 tabular 設定下，$Q(s,a)$ 是一個獨立的純量，$\nabla_w Q(s,a;w) = \mathbf{e}_{s,a}$（one-hot），就還原回 Q-learning 的更新式。

> **Note：** 這裡的 semi-gradient 做法（target 不對 $w$ 求導）並不是數學上的真正梯度，但它讓更新穩定。
> 若連 target 也一起對 $w$ 求導，結果是 residual gradient algorithm，收斂性質更弱，實務上幾乎不用。

---

## 為什麼 Q-learning 能收斂？

### 收斂定理與必要條件

**定理（Watkins & Dayan, 1992）**：在以下條件下，tabular Q-learning 以機率 1 收斂到 $Q^*$：

**條件 1：每個 state-action pair 被無限次訪問**

$$
\sum_{t=1}^{\infty} \mathbf{1}[(s_t, a_t) = (s, a)] = \infty \quad \forall (s,a)
$$

**條件 2：學習率滿足 Robbins-Monro 條件**

$$
\sum_{t=1}^{\infty} \alpha_t(s,a) = \infty \qquad \text{（步伐不能太快收斂）}
$$

$$
\sum_{t=1}^{\infty} \alpha_t(s,a)^2 < \infty \qquad \text{（累積噪聲要有限）}
$$

**條件 3：reward 有界**，即 $|R(s,a)| \le R_{\max}$。

如果行為策略採用 $\epsilon$-greedy，還要額外滿足一個常見的實務版本：$\epsilon_t \to 0$，但衰減不能太快，否則某些 state-action pair 可能還沒被充分探索就已經變成近乎 greedy。這就是常見的 GLIE（Greedy in the Limit with Infinite Exploration）想法：前期要探索，後期才逐漸變貪婪，但整體仍要保證每個動作被看到無限次。

### 為什麼這兩個條件缺一不可？

從隨機近似（stochastic approximation）的角度來看，Q-learning 本質上是用帶噪聲的樣本，迭代地求解不動點的方程：

$$
Q(s,a) = \mathcal{T}^* Q(s,a)
$$

**第一個 $\sum \alpha = \infty$ 的必要性：**

若 $\sum \alpha_t(s,a) < \infty$，那麼不管學多久，Q 值能修正的總幅度有上限。若初始值離 $Q^*$ 太遠，永遠無法完全修正過來。

**第二個 $\sum \alpha^2 < \infty$ 的必要性：**

更新項 $\alpha_t \delta_t$ 帶有隨機噪聲（因為 $s_{t+1}$ 是隨機的）。這個噪聲的累積方差正比於 $\sum \alpha_t^2$。若 $\sum \alpha^2 = \infty$，噪聲會不斷疊加，Q 值無法穩定。

**最典型的合法選擇：**

$$
\alpha_t = \frac{1}{t} \quad \Rightarrow \quad \sum_t \frac{1}{t} = \infty,\quad \sum_t \frac{1}{t^2} = \frac{\pi^2}{6} < \infty \quad \checkmark
$$

但實務上常用固定 $\alpha$（如 $0.001$）。這樣 $\sum \alpha^2 = \infty$，嚴格說不滿足條件，但可以讓模型持續適應非穩態環境，是一種偏差-方差的工程取捨。

---

## 探索：學習的前提，不只是配角

### Greedy 為什麼不夠？

想像只有一個狀態 $s$，兩個動作 $A$（真實期望回報 $0.7$）與 $B$（真實期望回報 $0.5$）。

如果最一開始，第一次試 $A$ 拿到 $0$，第一次試 $B$ 拿到 $1$，此時：
 
$$
Q(s, A) = 0, \quad Q(s, B) = 1
$$

Greedy 策略此後永遠選 $B$，再也沒有機會修正 $Q(s, A)$，最終收斂到次優。

### 探索的必要性：從收斂條件看

條件 1 要求每個 $(s,a)$ 被無限次訪問，這直接決定了行為策略（behavior policy）不能是純 greedy。**探索不只是工程小技巧，它是收斂定理的前提**。

### $\epsilon$-greedy：最常見的折衷方案

$$
\pi(a \mid s) =
\begin{cases}
1 - \epsilon + \dfrac{\epsilon}{|\mathcal{A}|}, & a = \arg\max_{a'} Q(s, a') \\[6pt]
\dfrac{\epsilon}{|\mathcal{A}|}, & \text{otherwise}
\end{cases}
$$

實務上常讓 $\epsilon$ 從較大的值（如 $1.0$）隨訓練線性或指數衰減到某個小值（如 $0.05$），讓 agent 在早期充分探索，後期趨向 exploitation。

### Softmax 策略：讓信心強度影響選擇

$$
\pi(a \mid s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}
$$

溫度 $\tau$ 控制分佈的平坦程度：

- $\tau \to 0$：集中在最大 Q 值，等同 greedy。
- $\tau \to \infty$：接近均勻分佈，完全隨機。

$\epsilon$-greedy 把「選哪個」和「要不要探索」切開；softmax 則讓探索隨 Q 值的差距自適應——Q 值差異大時自然更傾向 greedy，差異小時更願意嘗試。

---

## Off-Policy 的本質：行為策略和目標策略可以不同

### 定義

- **On-policy**：用來收資料的策略 $\mu$，和正在被改進的策略 $\pi$ 是同一個。
- **Off-policy**：$\mu \ne \pi$。Q-learning 就是 off-policy：target 永遠是 greedy，但收資料可以用任何 $\mu$。

### Q-learning 為什麼是 off-policy？

target 裡的 $\max_{a'}$ 就是 greedy policy 的定義：

$$
y_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')
$$

不管 $a_t$ 是怎麼被選出來的（greedy、$\epsilon$-greedy、甚至人類操作），更新的目標都是朝 Bellman optimality equation 逼近。

對比 **SARSA**（on-policy 版本），它用**實際下一步動作** $a_{t+1}$ 估計 target：

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl(r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)\bigr)
$$

因為 $a_{t+1}$ 是從當前策略抽到的，所以 SARSA 學到的是「當前策略的價值」，而非最優策略的價值。

| | Q-learning | SARSA |
|---|---|---|
| Target 的 next-action | $\max_{a'} Q(s_{t+1}, a')$ | $Q(s_{t+1}, a_{t+1})$（實際採樣） |
| 本質 | Off-policy | On-policy |
| 學到的是 | $Q^*$（最優） | $Q^\pi$（當前策略） |
| 適合場景 | 可重用過去資料 | 需要安全的探索策略 |

Off-policy 的最大實務優勢：可以把**別的策略**（舊版模型、人類示範、保守控制器）收集的資料拿來學，大幅提高 data efficiency。

---

## Overestimation Bias：max 運算天生偏樂觀

### 數學上為什麼會高估？

設有一組帶雜訊的 Q 估計 $\hat{Q}(s', a')$，真實值為 $Q^*(s', a')$，誤差是零均值隨機變數 $\epsilon_{a'}$：

$$
\hat{Q}(s', a') = Q^*(s', a') + \epsilon_{a'}
$$

比較 max 的期望值：

$$
\mathbb{E}\!\left[\max_{a'} \hat{Q}(s', a')\right] \ge \max_{a'} \mathbb{E}[\hat{Q}(s', a')] = \max_{a'} Q^*(s', a')
$$

第一個不等式來自 Jensen's inequality（$\max$ 是凸函數）。這說明：**只要估計值帶有雜訊，取 max 就會系統性高估**。

### 一個具體例子

假設只有兩個動作，$Q^*(s', A) = Q^*(s', B) = 0$，但估計值有均勻噪聲 $\epsilon \sim U[-1, 1]$：

$$
\mathbb{E}\!\left[\max(\hat{Q}_A, \hat{Q}_B)\right] = \mathbb{E}[\max(\epsilon_A, \epsilon_B)] = \frac{1}{3} > 0
$$

真實最大值是 $0$，但期望估計卻是 $\frac{1}{3}$。這個 bias 會隨著動作數量增多而惡化，也會隨 Q 值本身的噪聲放大而放大。

### Double Q-learning：把「選」和「評」拆開

核心思想：用一組 $Q^A$ 決定哪個動作最好，再用另一組 $Q^B$ 評估那個動作的值：

$$
y_t = r_{t+1} + \gamma Q^B\!\left(s_{t+1},\; \arg\max_{a'} Q^A(s_{t+1}, a')\right)
$$

更新 $Q^A$：

$$
Q^A(s_t,a_t) \leftarrow Q^A(s_t,a_t) + \alpha\bigl[y_t - Q^A(s_t,a_t)\bigr]
$$

下一輪交換 $A$、$B$ 的角色。

**為什麼有效？** 若選動作和評估動作的誤差是相對獨立的，那麼「挑到噪聲恰好很大的動作，再用另一個估計評它」的機率降低，高估傾向自然減弱。

---

## 從 Tabular 到神經網路：為什麼會不穩定？

### 狀態空間的詛咒

Atari 的原始影像狀態空間，不同 frame 的組合數量遠超出可存表格的範圍；圍棋的狀態空間約 $10^{170}$，甚至更大。Tabular Q-learning 根本無法擴展。

解法是用帶參數 $w$ 的函數近似：

$$
\hat{Q}(s, a; w) \approx Q^*(s, a)
$$

通常就是神經網路。此時更新式從對單一表格格子的修改，變成對整個網路做梯度下降。

### 為什麼直接套上神經網路會崩？

**問題一：樣本高度相關。**

Agent 在環境裡順序跑，得到的軌跡 $(s_t, a_t, r_{t+1}, s_{t+1})$ 是高度相關的時間序列。SGD 隱含的 IID 假設被嚴重違反，梯度估計的方差極高，訓練容易震盪。

**問題二：Target 隨著參數移動。**

Target $y_t = r_{t+1} + \gamma \max_{a'} \hat{Q}(s_{t+1}, a'; w)$ 也依賴 $w$，等於每次梯度更新都同時在移動目標。這不像監督學習有固定的標籤——標籤會跟著模型一起動。這種 **moving target** 讓訓練的穩定性遠低於普通監督學習。

DQN 的兩個核心裝置分別對準這兩個問題。

---

## DQN：兩個工程裝置如何救穩定性

### 損失函數

DQN 的訓練目標：

$$
\mathcal{L}(w) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\!\left[\frac{1}{2}\Bigl(r + \gamma \max_{a'} \hat{Q}(s',a';\bar{w}) - \hat{Q}(s,a;w)\Bigr)^2\right]
$$

關鍵：target 用的是 $\bar{w}$（target network），不是 $w$（online network）。

這個設計等於把原本會跟著 $w$ 一起移動的 target，改成一個慢速變動的目標。從優化角度看，online network 負責快速吸收新樣本，target network 則提供一個較平穩的參考座標系，避免「追著自己剛改過的答案跑」。

梯度對 $w$ 求（視 $\bar{w}$ 為常數）：

$$
\nabla_w \mathcal{L} = -\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\!\left[\delta \cdot \nabla_w \hat{Q}(s,a;w)\right]
$$

其中 $\delta = r + \gamma \max_{a'} \hat{Q}(s',a';\bar{w}) - \hat{Q}(s,a;w)$。

### 裝置一：Experience Replay

每次互動的 transition 存入 replay buffer $\mathcal{D}$，訓練時從 $\mathcal{D}$ 隨機抽 mini-batch。

**效果 1：打破相關性。**
隨機抽樣讓 mini-batch 的樣本分布接近 IID，梯度估計更穩定。

**效果 2：提高資料利用率。**
同一筆 transition 可以在不同階段被多次取用，不像 on-policy 方法每筆資料只用一次。

**Buffer 設計細節：**
通常用定容的 FIFO 佇列（如 $10^6$ 筆），新資料覆蓋最舊的記錄。Buffer 要夠大到讓樣本分布「洗勻」，又不能太大到讓太舊的資料主導訓練。

### 裝置二：Target Network

維持兩組參數：

- **Online network** $w$：每步用梯度更新。
- **Target network** $\bar{w}$：每隔固定步數（如 $10{,}000$ 步）從 $w$ 硬複製過來，其餘時間固定不動。

Target 固定住 $C$ 步之內，類似短期內把「答案紙」鎖起來，讓模型有安全的目標可以追。隔一段時間再把最新的估計更新進去。

**另一種做法：soft update（Polyak averaging）**

$$
\bar{w} \leftarrow \tau w + (1-\tau) \bar{w}, \quad \tau \ll 1
$$

Target 緩慢追蹤 online network，分佈改變更平滑，常見於 Actor-Critic 類方法（如 DDPG、TD3、SAC）。

### DQN 完整訓練流程

```
初始化：
  online network Q(s,a;w)
  target network Q(s,a;w̄) = Q(s,a;w)
  replay buffer D（空的）

for episode = 1, 2, ...:
  s = env.reset()
  for t = 1, 2, ...:
    用 ε-greedy 選動作 a
    執行 a，觀察 r, s'
    存入 D: (s, a, r, s')

    if len(D) >= batch_size:
      從 D 抽 mini-batch B
      for (s_i, a_i, r_i, s'_i) in B:
        if s'_i 是 terminal:
          y_i = r_i
        else:
          y_i = r_i + γ * max_a' Q(s'_i, a'; w̄)

      L = Σ (y_i - Q(s_i, a_i; w))^2 / 2
      w ← w - α ∇_w L

    每 C 步：w̄ ← w
    s = s'
```

---

## 進階技巧：修掉 DQN 的四個痛點

### Double DQN

把上面介紹的 Double Q-learning 搬到神經網路：

$$
y = r + \gamma \hat{Q}\!\left(s',\; \arg\max_{a'} \hat{Q}(s', a'; w);\; \bar{w}\right)
$$

用 online network 選最優動作，用 target network 評估那個動作。相比 DQN 的 $\max_{a'} \hat{Q}(s', a'; \bar{w})$，只改了一行，效果卻顯著：Overestimation 大幅減少。

### Prioritized Experience Replay（PER）

直覺：TD error 大的樣本，代表模型在那個 transition 上還學得不好，應該更常被抽到。

定義採樣機率：

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \qquad p_i = |\delta_i| + \epsilon
$$

其中 $\alpha \in [0,1]$ 控制優先程度（$\alpha=0$ 等同均勻抽樣）。

由於這樣改變了資料分佈，梯度估計引入了偏差，需要用 Importance Sampling 權重修正：

IS weight:

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta
$$

這個修正的目的，是把「被優先抽到」所造成的偏差拉回來。若某筆樣本被抽中的機率越大，就代表它在梯度中的影響應該被相對縮小；反過來，稀有樣本的影響則要放大，才能讓整體更新仍然接近原始的期望目標。

訓練時把每筆樣本的 loss 乘上對應的 IS 權重。$\beta$ 從 $0$ 逐漸退火到 $1$，訓練初期讓偏差存在（此時估計本身就很差），後期消除偏差。

### Multi-step Return

不只看一步，而是展開 $n$ 步：

$$
G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n \max_{a'} \hat{Q}(s_{t+n}, a')
$$

這個式子其實是在把 Bellman backup 往前展開 $n$ 步：先真的累積 $n$ 步內看得見的 reward，再把更遠處的部分交給 bootstrap estimate。當 $n=1$ 時，它退化回標準的 one-step Q-learning；當 $n$ 變大時，真實回饋會更早傳回來，但估計方差也會一起變大。

Target 變成：

$$
y_t = G_t^{(n)}
$$

**偏差-方差的 trade-off：**
- $n=1$（單步）：高偏差（bootstrap estimate 可能跑偏），低方差。
- $n=T$（完整 episode）：零偏差，但高方差（對整段隨機軌跡求和）。
- 通常 $n=3\sim5$ 效果最好，讓真實 reward 更快傳遞到更早的 state。

### Dueling Networks

把 Q 函數分解成：

$$
\hat{Q}(s,a;w) = \hat{V}(s;w_V) + \left[\hat{A}(s,a;w_A) - \frac{1}{|\mathcal{A}|}\sum_{a'} \hat{A}(s,a';w_A)\right]
$$

其中 $\hat{V}$ 是狀態價值，$\hat{A}$ 是 advantage（選不同動作帶來的額外收益）。

**為什麼要減掉平均 advantage？**

若不減，$\hat{V}$ 和 $\hat{A}$ 的分解不唯一——你可以把任何常數加到 $\hat{V}$、減去 $\hat{A}$，$Q$ 不變。減掉平均之後，對任何 $s$，advantage 的均值永遠是 $0$，分解變得唯一（identifiable）。

**什麼時候最有效？** 當動作的選擇對結果影響不大時（例如很多遊戲的空白走廊），$\hat{A}$ 接近 $0$，模型只需要學好 $\hat{V}$，不需要為每個動作都估計一遍。

---

## 連續動作空間：max 運算的瓶頸

DQN 的架構假設動作空間是**有限的離散集合**——因為 $\max_{a'} Q(s', a')$ 需要枚舉所有動作。連續動作空間（如機器手臂的關節角度）無法直接適用。

### 方法一：離散化

最簡單暴力：把每個維度切成格子。

問題：若動作有 $d$ 個維度，每維度 $k$ 個格子，總動作數是 $k^d$，指數爆炸。低維（$d \le 3$）時勉強可用，高維不可行。

### 方法二：Normalized Advantage Function（NAF）

假設 advantage 是動作的二次型：

$$
A(s,a;w) = -\frac{1}{2}(a - \mu(s;w_\mu))^T P(s;w_P)(a - \mu(s;w_\mu))
$$

其中 $P(s;w_P)$ 是正定矩陣（透過 Cholesky 分解保證正定性）。

這個假設的關鍵是：一旦 advantage 對動作是凹的二次型，$Q$ 對動作的最大值就可以直接解析地求出，不需要再做昂貴的連續優化。換句話說，NAF 是用很強的結構假設，換掉連續動作空間裡那個最麻煩的 $\arg\max$。

此時：

$$
\max_a Q(s,a;w) = Q(s,a=\mu(s;w_\mu);w) = V(s;w_V)
$$

最優動作閉合解就是 $\mu(s;w_\mu)$，不需要額外搜尋。但代價是限制了 Q 函數的形狀——quadratic advantage 是很強的結構假設。

### 方法三：Amortized Q-learning

訓練一個 proposal network $\mu(a|s)$，生成候選動作集合，再從中取最大 Q 值近似 max：

$$
\max_{a \in \mathcal{A}} Q(s,a;w) \approx \max_{a \sim \mu(\cdot|s)} Q(s,a;w)
$$

把「在全空間搜尋」轉成「在有希望的候選集合內搜尋」，大幅降低計算成本，同時保留更靈活的 Q 函數形式。

---

## Value-Based RL 的演化脈絡

把這一篇梳理的方法串起來看，它是一條清晰的修補鏈：

```
Q-learning（樣本近似 Bellman update）
    ↓ 狀態空間太大，表格無法存
函數近似 Q-learning（神經網路參數化）
    ↓ 訓練不穩：相關樣本 + moving target
DQN（Experience Replay + Target Network）
    ↓ 高估偏差
Double DQN（選動作/評動作分開）
    ↓ 資料利用不均勻
PER（依 TD error 優先採樣）
    ↓ bootstrap 偏差 vs 方差 trade-off
Multi-step Return
    ↓ Q 函數形式可以更有結構
Dueling Networks
    ↓ 動作連續，無法 argmax
NAF / Amortized Q-learning
```

每一步都在回答：「現在最大的瓶頸在哪？怎麼修？」

---

## 常見誤解

**誤解 1：Q-learning 只要收資料夠多，一定能收斂**

不是。收斂定理有明確的前提：每個 $(s,a)$ 必須被無限次訪問，學習率必須滿足 Robbins-Monro 條件。只是「跑很久」，如果探索不充分或學習率選錯，一樣可能不收斂。

**誤解 2：DQN 的 Experience Replay 只是提高資料效率**

它同時也在破壞樣本相關性，讓 mini-batch 更接近 IID，解決的是訓練穩定性問題，而非只是 data reuse。

**誤解 3：Target Network 讓 target 固定，所以是「正確的標籤」**

不是。Target Network 的輸出仍然是帶有誤差的 bootstrap estimate，只是暫時固定住，讓 online network 有穩定的追蹤目標。它不是 oracle，只是穩定訓練的工程技巧。

**誤解 4：Dueling Networks 把 V 和 A 分開，所以模型能解釋哪些動作重要**

分解是網路架構上的設計，不代表學出來的 $\hat{V}$ 和 $\hat{A}$ 一定對應語意上的「狀態價值」和「動作優勢」。可解釋性需要額外驗證，不是自動得到的。

---

## 總結

1. **Q-learning 是用樣本近似 Bellman optimality update**。TD error $\delta_t$ 是估計的差距，每次更新縮小一小步，等同在做帶噪聲的不動點迭代。

2. **收斂的前提是每個 state-action pair 被充分探索，且學習率滿足 Robbins-Monro 條件**。探索策略（$\epsilon$-greedy、softmax）不只是工程選擇，而是理論收斂的保證。

3. **Q-learning 是 off-policy 的**：行為策略負責收資料，target 始終是 greedy。這讓歷史資料或其他策略的資料也能拿來學。

4. **$\max$ 運算天生帶有高估偏差**，來自 Jensen's inequality。Double Q-learning 把選動作和評動作拆開，是最直接的修正方式。

5. **DQN 的 Experience Replay 解決樣本相關性，Target Network 解決 moving target**，這兩個裝置讓函數近似版的 Q-learning 在實務上可訓練。

6. **連續動作空間需要額外設計**：NAF 用二次型假設換取閉合解，Amortized Q-learning 則用 proposal network 縮減搜尋空間，各有代價。

7. 整條演化鏈的邏輯只有一個：**找到當下最大的瓶頸，針對性地修補**，從 tabular 到 DQN 再到後續的改進，都是這個思路的延伸。
