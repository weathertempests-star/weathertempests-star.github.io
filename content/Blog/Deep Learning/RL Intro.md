---
type: Blog-post
tags:
  - Deep-Learning
  - note
  - Reinforcement-Learning
created: 2026-05-27
updated: 2026-05-27
authors: WeiTa
status: active
---

---



## 這篇文章想回答什麼問題？

前幾篇文章談的模型，不管是 MLP、CNN 還是 Transformer，都有一個共同前提：

> 你手上有一批資料，每筆資料都附有標籤。模型的任務，就是把輸入映射到正確的輸出。

但現實中很多問題根本不長這樣。

你不可能在訓練 AlphaGo 時，替每一步棋都標上「這步好」或「這步不好」。你也不可能在訓練自駕車時，為每個方向盤動作都準備答案。
這些任務裡，唯一能拿到的訊號，是和環境互動之後得到的**回饋**，而且這個回饋通常是延遲的、稀疏的、甚至容易出錯的。

Reinforcement Learning 要處理的，就是這個情境。

這篇文章想回答的四個問題：

1. RL 和 supervised learning 的本質差別在哪裡？
2. 怎麼用數學語言把「決策問題」寫清楚？
3. Bellman 方程在說什麼，為什麼它這麼重要？
4. 從 Bellman 方程到實際演算法，中間發生了什麼？

如果把這篇濃縮成一句話：

> RL 不是在學「預測標籤」，而是在學「如何在不確定的環境中，透過互動最大化長期回報」。

---

## 為什麼 Supervised Learning 不夠用？

### 標籤是一種奢侈

讓我們先把 supervised learning 的基本假設說清楚：

$$
\mathcal{D} = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(N)}, y^{(N)})\}
$$

每筆資料 $(x, y)$ 都是獨立、同分佈地從某個固定的 $p(x, y)$ 抽出來的。模型的目標是學出一個函數 $f_\theta$，讓預測結果盡量靠近標籤。

這個設定的隱藏前提是：**資料是被動收集的，而且你的動作不會影響接下來看到的資料。**

RL 的世界完全不是這樣。

| 面向 | Supervised Learning | Reinforcement Learning |
|---|---|---|
| 監督訊號 | 明確標籤 $y$ | reward $r$（通常稀疏且延遲） |
| 資料假設 | IID | 強烈 non-IID |
| 資料來源 | 被動收集 | 主動探索環境 |
| 優化目標 | 單筆預測正確 | 整段軌跡的長期總回報最大化 |
| 動作影響 | 當前輸出不影響下一筆輸入 | 當前動作直接決定下一個 state |

最後一行是關鍵。在 RL 裡，**你的選擇會改變你接下來看到的世界**。這讓整個問題的難度跳升了一個層次。

### Reward 不是答案本身

RL 裡的學習訊號叫做 reward，但它和標籤最大的差別是：

> reward 只是目標的 proxy，不是目標本身。

來看幾個設計例子就能理解問題在哪：

- 賽車遊戲：把「車速快」設為 reward，結果 agent 可能學會在固定角落不停旋轉，因為轉圈的速度夠高。
- 機器翻譯：逐字給 reward，但片段正確不代表整句通順。
- 推薦系統：以點擊率為 reward，結果模型可能學會推激情標題，而非真正有價值的內容。

這種現象叫 **reward hacking**，是 RL 在實務上最棘手的問題之一，也是設計 reward 時最容易踩的地雷。

所以開始之前，第一件事不是想演算法，而是想清楚：

> 我真正想要的行為是什麼？這個 reward 能不能準確代理那個目標？

---

## 核心想法：把決策問題寫成數學

### 從互動到 Trajectory

RL 裡 agent 和環境的互動，可以用一段 **trajectory** 來記錄：

$$
\mathcal{T} = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_t, a_t, r_{t+1}, \dots)
$$

每個時間步發生的事情：

1. Agent 觀察到當前 state $s_t$。
2. 根據策略選擇動作 $a_t$。
3. 環境給出 reward $r_{t+1}$，並轉移到下一個 state $s_{t+1}$。

這裡有一個細節值得注意：**state 和 observation 不一定相同。**

- **state**：環境的真實內部狀態（完整資訊）。
- **observation**：agent 實際看得到的資訊（可能是 state 的部分）。

如果 observation 不等於 state，這個問題就叫 Partially Observable MDP (POMDP)，難度大幅上升。這篇先假設 **fully observable**，讓問題可以處理。

---

## 模型框架：Markov Decision Process

要把這個問題寫成可解的數學形式，需要把環境結構化。最常用的框架是 **Markov Decision Process（MDP）**：

$$
\mathcal{M} = \langle \mathcal{S},\ \mathcal{A},\ P,\ R,\ \gamma \rangle
$$

- $\mathcal{S}$：狀態空間
- $\mathcal{A}$：動作空間
- $P(s' \mid s, a)$：狀態轉移機率
- $R(s, a)$：即時 reward 函數
- $\gamma \in [0,1)$：折扣因子

### Markov Property：把歷史壓進 State

MDP 的核心假設是：

$$
\mathbb{P}(s_{t+1} \mid s_t, a_t) = \mathbb{P}(s_{t+1} \mid s_0, a_0, s_1, a_1, \dots, s_t, a_t)
$$

白話版：**下一個 state，只取決於當前 state 和當前動作，跟更早的歷史無關。**

這不是在說歷史不重要，而是在說：如果 state 定義得夠好，歷史的影響應該已經被壓縮進 state 裡了。好的 state 設計，正是讓 Markov property 成立的關鍵。

### 策略 Policy：Agent 怎麼做決定

策略 $\pi$ 描述 agent 在每個 state 下選動作的方式：

$$
\pi(a \mid s) := \mathbb{P}(a_t = a \mid s_t = s)
$$

- **Deterministic policy**：$\pi(s) = a$，直接給出唯一動作。
- **Stochastic policy**：對所有動作給出機率分佈。

RL 的學習目標，就是找到一個策略 $\pi^*$，讓 agent 的長期回報最大化。

---

## 數學推導

### Return：我們到底在最大化什麼？

RL 不看單步 reward，而是看整段未來累積回報，稱為 **return**：

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

折扣因子 $\gamma$ 控制「多在乎未來」：

- $\gamma \to 1$：未來和現在一樣重要，agent 有遠見。
- $\gamma \to 0$：只在乎下一步，目光短淺。

折扣的存在有兩個作用：數學上讓無窮級數收斂，語意上則表達「未來越遠越不確定，折扣是對這份不確定性的懲罰」。

> Note：return 有一個非常有用的遞迴關係：
> $$G_t = r_{t+1} + \gamma G_{t+1}$$
> 這個看起來簡單的分拆，正是 Bellman 方程的起點。

---

### Value Function：把長期回報壓成一個數字

直接比較不同策略的 trajectory 很難，因為軌跡是隨機的。我們需要一個穩定的量來代表「這個 state 或 action 到底有多好」。

**State value function**：

$$
V^{\pi}(s) := \mathbb{E}_{\pi}\left[G_t \mid s_t = s\right]
= \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s\right]
$$

它回答的問題是：從 state $s$ 出發，照策略 $\pi$ 走，平均能拿多少 return？

**Action value function（Q function）**：

$$
Q^{\pi}(s, a) := \mathbb{E}_{\pi}\left[G_t \mid s_t = s,\ a_t = a\right]
$$

它回答的問題是：在 state $s$ 先執行動作 $a$，再照策略 $\pi$ 走，平均 return 是多少？

兩者的關係很直觀：

$$
V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s)\ Q^{\pi}(s, a)
$$

也就是說，$V^{\pi}(s)$ 是 $Q^{\pi}(s, a)$ 在策略 $\pi$ 下的期望值。

---

### Bellman Expectation Equation：遞迴是關鍵

把 $G_t = r_{t+1} + \gamma G_{t+1}$ 代入 value function，可以把長期問題拆成兩部分：

$$
V^{\pi}(s) = \mathbb{E}_{\pi}\left[r_{t+1} + \gamma G_{t+1} \mid s_t = s\right]
$$

展開期望值（對動作、對下一個 state 各自積分）：

$$
\boxed{V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^{\pi}(s') \right]}
$$

這就是 **Bellman Expectation Equation**，對應的 Q function 版本是：

$$
Q^{\pi}(s, a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s')\ Q^{\pi}(s', a')
$$

這個方程式說的事情很清楚：

> **當前的 value = 立即 reward + 折扣後的下一個 state value 的期望**

它把一個無窮的求和問題，改寫成了一個遞迴方程式。

#### Bellman Operator 的視角

把 Bellman equation 改寫成算子形式，能更清楚地看到它的結構。定義 Bellman expectation operator $\mathcal{T}^{\pi}$：

$$
(\mathcal{T}^{\pi} V)(s) = \sum_{a} \pi(a \mid s) \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\ V(s') \right]
$$

那麼 Bellman equation 就是說：

$$
V^{\pi} = \mathcal{T}^{\pi} V^{\pi}
$$

也就是說，**$V^{\pi}$ 是算子 $\mathcal{T}^{\pi}$ 的不動點（fixed point）。**

可以證明，當 $\gamma < 1$ 時，$\mathcal{T}^{\pi}$ 是一個**收縮映射（contraction mapping）**，對任意兩個值函數 $V$、$V'$：

$$
\|\mathcal{T}^{\pi} V - \mathcal{T}^{\pi} V'\|_{\infty} \le \gamma \|V - V'\|_{\infty}
$$

這保證了從任意初始值出發，反覆套用 $\mathcal{T}^{\pi}$ 都會收斂到唯一的不動點 $V^{\pi}$。這也是 iterative policy evaluation 能成立的數學根基。

---

### Bellman Optimality Equation：找最好的策略

如果不只想評估某個策略，而是要找最優策略，就定義：

$$
V^*(s) := \max_{\pi} V^{\pi}(s), \qquad Q^*(s,a) := \max_{\pi} Q^{\pi}(s,a)
$$

對 $V^*$ 套用一樣的遞迴邏輯，但把「依策略加權」換成「取最大值」：

$$
\boxed{V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\ V^*(s') \right]}
$$

對應的 Q function 版本：

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \max_{a'} Q^*(s', a')
$$

這個方程式之所以重要，是因為它直接告訴你最優策略長什麼樣：

$$
\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s, a)
$$

**一旦知道 $Q^*$，最優策略只需要每步 greedy 選最大 Q 值的動作。**

Bellman optimality operator $\mathcal{T}^*$ 同樣是收縮映射，收斂至唯一不動點 $V^*$，這正是 value iteration 的理論保證。

---

## 從方程式到演算法：三條路

### 1. Iterative Policy Evaluation

**設定：知道策略 $\pi$，要算出 $V^{\pi}$。**

把 Bellman equation 當更新規則來迭代：

$$
V_{k+1}(s) \leftarrow \sum_{a} \pi(a \mid s) \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\ V_k(s') \right]
$$

由收縮映射定理保證，$V_k \to V^{\pi}$。這是 Policy Iteration 的核心子步驟。

### 2. Policy Iteration

**設定：知道環境模型（$P$、$R$），要找最優策略 $\pi^*$。**

交替做兩件事，直到策略不再改變：

**Step 1 — Policy Evaluation**：計算 $V^{\pi_k}$。

**Step 2 — Policy Improvement**：更新為 greedy 策略：

$$
\pi_{k+1}(s) = \arg\max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\ V^{\pi_k}(s') \right]
$$

#### 為什麼 Policy Improvement 真的會變好？

從 value function 的定義出發：

$$
V^{\pi_k}(s) = \sum_a \pi_k(a \mid s)\ Q^{\pi_k}(s,a) \le \max_a Q^{\pi_k}(s,a) = Q^{\pi_k}(s, \pi_{k+1}(s))
$$

第一步不等式成立，是因為加權平均不超過最大值。把 $Q$ 展開，右邊正是 $(\mathcal{T}^{\pi_{k+1}} V^{\pi_k})(s)$。因此有：

$$
V^{\pi_k} \le \mathcal{T}^{\pi_{k+1}} V^{\pi_k} \le (\mathcal{T}^{\pi_{k+1}})^2 V^{\pi_k} \le \cdots \to V^{\pi_{k+1}}
$$

最後一個箭頭收斂，是因為 $\mathcal{T}^{\pi_{k+1}}$ 是收縮映射。結論：

$$
V^{\pi_k}(s) \le V^{\pi_{k+1}}(s) \quad \forall s
$$

**每次 greedy improvement，策略一定不會變差。** 這是 policy iteration 能保證單調改善的數學根基。

### 3. Value Iteration

**設定：知道環境模型，直接逼近 $V^*$，不顯式維護策略。**

每次更新直接套 Bellman optimality operator：

$$
V_{k+1}(s) \leftarrow \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\ V_k(s') \right]
$$

這相當於把 policy evaluation 和 policy improvement 合併成一步。$V_k \to V^*$，最後從 $V^*$ 反推 $\pi^*$。

---

## 當環境模型未知：Q-Learning

上面三個方法都需要知道完整的環境模型。但大多數真實問題裡，$P$ 和 $R$ 是不知道的。

這時只能靠和環境互動拿到的樣本 $(s_t, a_t, r_{t+1}, s_{t+1})$ 來學習。

**Q-Learning** 的核心想法是：用樣本近似 Bellman optimality update。

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \underbrace{\left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]}_{\text{TD error}\ \delta_t}
$$

### TD Error 的意義

$$
\delta_t = \underbrace{r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{target（走過這步後的最新估計）}} - \underbrace{Q(s_t, a_t)}_{\text{目前的舊估計}}
$$

TD error 就是「我原本的估計，和觀察到新訊息後的修正值，差了多少」。每次更新都讓目前估計往 target 靠近一小步（步長由 $\alpha$ 控制）。

> Insight：TD error 是整個 RL 演算法設計的共同語言。不管是後來的 DQN、Actor-Critic 還是 PPO，它們的核心更新機制都是在處理「如何更穩定、更有效率地利用 TD error」這個問題。

Q-Learning 的一個漂亮性質：它是 **off-policy** 的。也就是說，用來收集資料的行為策略（behavior policy），和正在被優化的目標策略（target policy），可以不同。這讓我們可以用歷史資料或探索性策略收集的樣本來更新 $Q^*$ 的估計。

---

## 訓練時實際發生什麼？

### Exploration vs Exploitation 的兩難

Q-Learning 有一個實作上必須面對的根本問題：

> 如果一直選 Q 值最高的動作，永遠不會知道其他動作有多好。但如果一直隨機探索，永遠學不到好策略。

這就是 **exploration vs exploitation** 的兩難。最簡單的解法是 $\epsilon$-greedy：

- 以機率 $1 - \epsilon$：選 Q 值最高的動作（exploitation）。
- 以機率 $\epsilon$：隨機選動作（exploration）。

訓練初期 $\epsilon$ 較大，讓 agent 多探索；隨著訓練進行，逐漸縮小 $\epsilon$，讓策略趨於貪婪。

### Model-Based 和 Model-Free 的分野

**Model-based RL**：先學一個環境模型 $\hat{P}(s' \mid s, a)$ 和 $\hat{R}(s, a)$，再在模型裡做 planning。
- 優點：如果模型夠準，sample efficiency 高。
- 缺點：模型誤差會在 planning 中被放大（model bias）。

**Model-free RL**：不顯式學習環境模型，直接從互動樣本估計 value 或 policy。
- 優點：對複雜環境更有彈性，不需要對環境建模。
- 缺點：通常需要更多和環境的互動（sample inefficient）。

一個很有畫面的類比：

> Model-based 是先學會「看地圖」，再規劃路線。
> Model-free 是在陌生城市裡靠走路試錯，慢慢熟悉哪條路好走。

---

## 常見誤解

### 誤解 1：Reward 越多越好

RL 優化的是累積 reward，不是讓每一步的 reward 盡量高。有時候短期的低 reward，是為了換取長期的更大回報。折扣因子 $\gamma$ 正是控制這個 trade-off 的關鍵參數。

### 誤解 2：Bellman 方程只是一個遞迴公式

Bellman 方程的本質是把無窮長的優化問題，拆解成「一步的決策 + 未來價值」的結構。收縮映射定理保證了這個遞迴可以被迭代求解，是 DP 和 RL 演算法設計的根本。

### 誤解 3：Q-Learning 就是找到正確 Q 值

Tabular Q-Learning 在小型問題上確實能收斂到 $Q^*$，但真實問題的 state space 通常大到無法用表格存。後續的 DQN 正是在解「如何用神經網路近似 Q function」的問題，代價是收斂性的保證變得更複雜。

### 誤解 4：探索不重要

如果 agent 從沒走到某些 state，它永遠不知道那些 state 有多好或多壞。Exploration 是 RL 能正確學習的前提。缺乏探索，訓練結果可能是一個「局部安全卻次優」的策略。

---

## 總結

1. RL 要解的問題和 supervised learning 本質不同：資料不是 IID 的，且 agent 的行動會直接影響接下來看到的世界。

2. MDP 提供了一個可處理的數學框架，把環境壓縮成 $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$。Markov property 是這個框架的核心假設。

3. Bellman equation 把無窮長的 return 拆成「一步 + 未來」，得到一個遞迴結構，並以 Bellman operator 的形式寫出來。收縮映射定理保證了迭代求解的收斂性。

4. $V^{\pi}$ 和 $Q^{\pi}$ 分別回答「這個 state 好不好」和「這個 action 好不好」。最優策略只需要對 $Q^*$ 做 greedy selection。

5. Policy iteration、value iteration、Q-learning，是在不同資訊條件下逼近最優解的三條路線；而 TD error 是後續深度 RL 演算法（DQN、Actor-Critic、PPO）的共同語言。

6. Reward design 和 exploration 策略，是 RL 實務上最容易出問題的地方，再好的演算法也逃不掉。

