---
type: research-notebook
tag:
  - Mamba
  - SSM
  - LLM
  - sequence-modeling
created: 2026-05-18
updated: 2026-05-21
authors: WeiTa
---

# Mamba-3: Improved Sequence Modeling using State Space Principles

---

## Research Questions

| # | Question | Answered By |
|---|----------|-------------|
| Q1 | Can SSM discretization itself provide local mixing without an explicit convolution layer? | Exponential-Trapezoidal → implicit convolution |
| Q2 | Can linear recurrent models recover state-tracking capability lost by real-valued diagonal parameterization? | Complex-valued SSM |
| Q3 | Can recurrent models increase arithmetic intensity during decoding without enlarging state size? | MIMO |
| Q4 | Do LTV SSMs have a principled discretization framework analogous to LTI methods? | Exponential-Adjusted Framework |

---

## Evolution of Mamba Series

| Version | Core Innovation | Problem Solved |
|---------|----------------|----------------|
| Mamba-1 | Selective SSM (data-dependent $\Delta, B, C$) | Content-based reasoning |
| Mamba-2 | SSD + scalar-$A$ parameterization | Hardware-efficient parallel training |
| Mamba-3 | Trapezoidal discretization + complex dynamics + MIMO | Expressivity + decoding efficiency |

**Design-space view across axes:**

| Axis | Mamba-1 | Mamba-2 | Mamba-3 |
|------|---------|---------|---------|
| Recurrence | Selective | SSD (Euler) | SSD (Trapezoidal) |
| Dynamics | Real diagonal | Real scalar | Complex diagonal |
| Local mixing | External conv | External conv | Implicit conv (absorbed) |
| Parallelism | Partial | Full SSD | Full SSD |
| Decoding intensity | ~2.5 ops/byte | ~2.5 ops/byte | $R\times$ higher (MIMO) |

---

## Background

### Transformer Bottleneck

Transformers carry $O(T^2)$ time and memory complexity in the attention mechanism. As context length grows, both inference latency and KV-cache memory become prohibitive, motivating sub-quadratic alternatives.

### Limitations of Prior Linear Models

Two persistent issues remain in existing SSM/linear-attention architectures:

1. **Expressivity gap**: Real-valued diagonal SSMs cannot represent rotational hidden-state dynamics, causing failures on state-tracking tasks (e.g., parity).
2. **Decoding inefficiency**: Training-optimized algorithms produce memory-bound decode steps with low arithmetic intensity (~2.5 ops/byte on Mamba-2 vs. ~295 ops/byte compute-bound on H100).

### Mamba-3 Solutions (Overview)

| Contribution                           | Mechanism                                       | Effect                                                                  |
| -------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------- |
| Exponential-Trapezoidal Discretization | Second-order integration scheme                 | Implicit convolution; replaces standalone conv layer                    |
| Complex-valued SSM                     | Complex diagonal $A$ → rotation in latent space | Restores state-tracking; equivalent to data-dependent RoPE              |
| MIMO                                   | Rank-$R$ state-input expansion                  | Trades FLOPs for memory bandwidth; higher decoding arithmetic intensity |
> 論文核心：Mamba-3 提出了三項主要的方法學改進，以解決過去線性模型在表達能力與推論效率上的瓶頸：
> 1. 指數梯形離散化（產生隱式卷積效果）
> 2. 複數狀態空間模型（恢復狀態追蹤與邏輯推理能力）
> 3. MIMO 架構（大幅提升解碼階段的硬體算術強度與運算效率)

---

## Recap: SSM Foundations

### Continuous-Time SSM

$$\dot{h}(t) = A(t)\,h(t) + B(t)\,x(t), \qquad y(t) = C(t)^\top h(t)$$

### Mamba-1/2: Exponential-Euler Discretization

Given step size $\Delta_t$, the state-input integral is approximated by Euler's rule (right-endpoint):

$$h_t = e^{\Delta_t A_t}\,h_{t-1} + \Delta_t B_t x_t, \qquad y_t = C_t^\top h_t$$

Mamba-2 constrains $A_t = a_t I_{N\times N},\; a_t \in \mathbb{R}_{<0}$ (scalar-times-identity), enabling efficient tensor-core computation. Defining $\alpha_t := e^{\Delta_t a_t} \in (0,1)$:

$$h_t = \alpha_t h_{t-1} + \Delta_t B_t x_t$$

**Operator view:** $\alpha_t$ is a scalar **decay operator** (memory horizon); $\Delta_t$ jointly scales both state-transition and state-input — a single scalar controls forgetting and input weighting simultaneously, which is an architectural constraint rather than a design choice.

### State Space Duality (SSD)

Mamba-2 showed that a wide class of SSMs admits a vectorized matrix form, establishing a duality between linear recurrence and masked attention:

$$Y = (L \odot C B^\top)\,X$$

where:
- $L \in \mathbb{R}^{T \times T}$: structured causal mask encoding decay
- $B, C \in \mathbb{R}^{T \times N}$: input/output projections (analogous to $K$, $Q$)
- $X \in \mathbb{R}^{T \times D}$: token inputs (analogous to $V$)

The mask $L$ for Mamba-2 is a **1-semiseparable matrix** — a lower-triangular matrix with entries $L_{ij} = \prod_{k=j+1}^{i} \alpha_k$ — scaled element-wise by $\Delta_t$:

$$L = \begin{bmatrix} 1 & & & \\ \alpha_1 & 1 & & \\ \alpha_1\alpha_2 & \alpha_2 & 1 & \\ \vdots & & & \ddots \end{bmatrix} \cdot \operatorname{Diag}(\Delta_1, \dots, \Delta_T)$$

This is the parallel (training-efficient) form; the recurrence is the sequential (inference-efficient) form. SSD makes both computationally accessible.

> 背景與動機：傳統 Transformer 的注意力機制在處理長文本時，運算與記憶體會呈平方增長 。
> 
> 先前的線性模型（如 Mamba-1/2）雖然改善了複雜度，但仍有兩大缺陷：首先是受限於實數對角矩陣的參數化，無法處理如「奇偶校驗」等需要狀態追蹤的任務
> 另外，這些模型在解碼 (decoding) 階段受到記憶體頻寬限制，硬體算術強度極低，導致運算資源閒置。

---

## Methodology

### 1. Exponential-Trapezoidal Discretization

#### Why Earlier Discretizations Fall Short

Prior SSM works derived discretization schemes (ZOH, Euler) for **linear-time invariant (LTI)** systems. Mamba-1 adapted ZOH to **linear-time varying (LTV)** systems without formal justification, relying on an additional heuristic. This lacks theoretical grounding and leaves accuracy on the table.

#### Exponential-Adjusted Framework

The framework derives discrete recurrences from the exact ODE solution, separating the state-transition integral (approximated via right-hand rule) from the state-input integral (where different methods diverge):

$$h_t \approx \underbrace{\exp(\Delta_t A_t)\,h_{t-1}}_{\text{state transition}} + \underbrace{\int_{\tau_{t-1}}^{\tau_t} \exp\!\bigl((\tau_t - \tau)A_t\bigr)\,B(\tau)\,x(\tau)\,d\tau}_{\text{state-input integral — choice of approximation}}$$

**Exponential-Euler** (Mamba-1/2): right-endpoint hold on the state-input integral → first-order, local truncation error $O(\Delta_t^2)$.

#### Exponential-Trapezoidal (Mamba-3)

Applies a data-dependent generalized trapezoidal rule to the state-input integral, evaluating both endpoints and interpolating:

$$\boxed{h_t = \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t}$$

| Parameter | Definition | Role |
|-----------|-----------|------|
| $\alpha_t$ | $\exp(\Delta_t A_t)$ | Decay operator |
| $\beta_t$ | $(1-\lambda_t)\,\Delta_t\,\exp(\Delta_t A_t)$ | Previous-token mixing weight |
| $\gamma_t$ | $\lambda_t\,\Delta_t$ | Current-token mixing weight |
| $\lambda_t$ | data-dependent $\in [0,1]$ | Interpolation parameter |

**Generalization**: $\lambda_t = 1$ recovers Mamba-2 (Euler); $\lambda_t = 1/2$ recovers classical trapezoidal. Error: $O(\Delta_t^3)$.

> 指數梯形離散化：過去的模型採用針對線性非時變 (LTI) 系統的近似方法，缺乏處理時變系統的嚴謹理論基礎。
> Mamba-3 引入了二階精度的梯形法則來處理積分，不僅降低了誤差，更引導出一個包含三個項目的遞迴式。
> 其中參數 $\lambda_t$ 可根據資料動態調整插值比例。

#### Implicit Convolution Interpretation

The three-term recurrence is equivalent to first applying a **width-2 data-dependent convolution** on the state-input $v_t = B_t x_t$, then running a standard linear recurrence:

$$v'_t = \beta_t v_{t-1} + \gamma_t v_t \quad \text{(conv on } B_t x_t\text{)}, \qquad h_t = \alpha_t h_{t-1} + v'_t$$

This is an **internal convolution** on projected inputs, unlike Mamba-1/2's external conv on raw $x_t$. Combined with learnable $B/C$ biases, it renders the standalone short causal convolution layer redundant.

> 隱式卷積角度：這個新的遞迴式在數學上等同於在輸入進入 SSM 核心之前，先套用一個寬度為 2 的資料依賴型卷積。
> 加上獨立的偏差值 (bias) 後，這種內部機制可以直接取代過去 Mamba 架構中必需的獨立短卷積層 (short causal convolution)。
#### SSD Parallel Form

The trapezoidal recurrence fits the SSD framework with a modified structured mask:

$$Y = (L \odot C B^\top)\,X$$

The mask $L$ becomes a **1-semiseparable + 2-band matrix** — the 2-band component encodes the implicit convolution weights $(\beta, \gamma)$, generalizing Mamba-2's diagonal.


#### Discretization Comparison Table

| Method | $\alpha_t$ | $\beta_t$ | $\gamma_t$ | Used In |
|--------|-----------|----------|-----------|---------|
| Forward Euler | $I + \Delta A$ | — | $\Delta$ | — |
| Backward Euler | $(I - \Delta A)^{-1}$ | — | $(I - \Delta A)^{-1}\Delta$ | — |
| Trapezoidal | $(I - \frac{\Delta}{2}A)^{-1}(I + \frac{\Delta}{2}A)$ | — | $(I - \frac{\Delta}{2}A)^{-1}\Delta$ | S4 |
| Zero-Order Hold | $\exp(\Delta A)$ | — | $A^{-1}(\exp(\Delta A)-I)$ | S4D, S5 |
| Exponential-Euler | $\exp(\Delta_t A_t)$ | — | $\Delta_t$ | Mamba-1/2 |
| **Exponential-Trapezoidal** | $\exp(\Delta_t A_t)$ | $(1-\lambda_t)\Delta_t\exp(\Delta_t A_t)$ | $\lambda_t\Delta_t$ | **Mamba-3** |

---

### 2. Complex-Valued SSM

#### Why Real-Valued Diagonal SSMs Fail

> **Core insight**: Eigenvalue geometry determines expressivity.

Real-valued diagonal SSMs can only **scale** hidden states along each dimension — the state transition is a contractive map. They cannot represent rotations in latent space.

This means real SSMs fundamentally cannot model periodic or phase-sensitive dynamics. A concrete failure: they cannot solve the parity task on binary sequences, which minimal finite automata handle trivially. Complex eigenvalues introduce $2\times 2$ rotation blocks, transforming the system from a **purely contractive** dynamical system into a **rotational** one.

#### Continuous Complex SSM

$$\dot{h}(t) = \operatorname{Diag}(A(t) + i\Theta(t))\,h(t) + (B(t) + i\hat{B}(t))\,x(t)$$

$$y(t) = \operatorname{Re}\!\bigl((C(t) + i\hat{C}(t))^\top h(t)\bigr)$$


#### Complex-to-Real Equivalence

Under exponential-Euler discretization, the complex state $h(t) \in \mathbb{C}^{N/2}$ maps to a real state $h_t \in \mathbb{R}^N$ (dimension doubling):

$$h_t = e^{\Delta_t A_t}\,R_t\,h_{t-1} + \Delta_t B_t x_t, \qquad y_t = C_t^\top h_t$$

with real-valued parameterization:
- $B_t := [B_t;\; \hat{B}_t] \in \mathbb{R}^N$
- $C_t := [C_t;\; -\hat{C}_t] \in \mathbb{R}^N$  
- $R_t := \operatorname{BlockDiag}(\{R(\Delta_t\theta_t[i])\}_{i=1}^{N/2}) \in \mathbb{R}^{N\times N}$ — block-diagonal of $2\times 2$ rotation matrices

#### Data-Dependent RoPE Equivalence (The RoPE Trick)

Unrolling the recurrence reveals equivalence with a scalar-transition SSM applying cumulative data-dependent rotations to $B$ and $C$:

$$h_t = e^{\Delta_t A_t}h_{t-1} + \left(\prod_{k=0}^{t} R_k^\top\right)\Delta_t B_t x_t$$

$$y_t = \left[\left(\prod_{k=0}^{t} R_k^\top\right)C_t\right]^\top h_t$$

The cumulative rotations $\prod R_k^\top$ are analogous to Rotary Position Embeddings (RoPE), applied to $B/C$ (i.e., $K/Q$ in the SSD attention view). This allows complex dynamics to be computed with minimal overhead: the real and imaginary parts decouple cleanly.

> 複數 SSM 與 RoPE 的等價性：
- 純實數的矩陣只能對狀態進行縮放，這導致模型缺乏「旋轉」動態的表達能力。
- Mamba-3 將狀態空間擴展至複數域，經過推導後證明，這在數學上完全等同於在標準的實數 SSM 中，對投影矩陣 $B$ 與 $C$ 套用資料依賴型的旋轉位置編碼 (RoPE Trick)。
- 這項改進以極低的運算成本解決了模型在狀態追蹤（如邏輯推理）上的根本缺陷 。
---

### 3. Multi-Input Multi-Output (MIMO)

#### Core Insight

> **MIMO trades FLOPs for memory bandwidth** — it increases arithmetic intensity during decoding without changing state size.

#### The Decoding Bottleneck

During autoregressive decode, each step reads the full hidden state $h_t \in \mathbb{R}^{N \times P}$ from memory and performs an outer product $B_t x_t^\top$, costing $\Theta(NP)$ FLOPs. Memory traffic dominates; arithmetic intensity is ~2.5 ops/byte — far below H100's compute-bound regime (~295 ops/byte). The hardware is almost entirely idle.

#### From SISO to MIMO

SISO recurrence for a head of dimension $P$:

$$h_t \leftarrow \alpha_t h_{t-1} + \Delta_t B_t x_t^\top, \quad B_t \in \mathbb{R}^N,\; x_t \in \mathbb{R}^P,\; h_t \in \mathbb{R}^{N\times P}$$

MIMO introduces rank $R$, expanding state-input terms:

$$B_t \in \mathbb{R}^N \;\to\; \mathbb{R}^{N\times R}, \quad x_t \in \mathbb{R}^P \;\to\; \mathbb{R}^{P\times R}, \quad C_t \in \mathbb{R}^N \;\to\; \mathbb{R}^{N\times R}$$

This converts the outer product into a **matrix-matrix multiply** — $R$ times more FLOPs at negligible additional memory traffic, elevating arithmetic intensity by $R\times$.

> MIMO 架構設計：模型在解碼時面臨嚴重的「記憶體牆」問題，計算單元都在等待資料載入。
> Mamba-3 透過引入 MIMO 架構，將輸入輸出維度擴展 $R$ 倍，成功把原本低效的「外積」運算轉換為 GPU 非常擅長的「矩陣相乘 (Matmul)」。
> 這代表模型能在不增加解碼延遲的狀況下，執行多出 $R$ 倍的有效運算量，大幅提升效能與硬體利用率。
#### Training: Chunked Algorithm

MIMO SSMs decompose as a sum of $R^2$ SISO SSMs:

$$y_t^{(i)} = \sum_{j=0}^{R-1} \operatorname{SSM}(\alpha, \Delta, B^{(j)}, C^{(i)}, x^{(j)})_t, \quad i \in \{0,\dots,R-1\}$$

Naive sequential compute costs $R\times$ overhead. With chunked training (chunk size $C$), setting $C_\text{MIMO} \leftarrow C_\text{SISO}/R$ preserves intra-chunk FLOP count, restricting training overhead to $R\times$ rather than $R^2\times$.

#### Parameter Control

Naive MIMO bloats $B,C,x,y,z$ projections by $R\times$. Mamba-3 mitigates this via:
- **Shared $B,C$** across heads (MVA structure): $DN \to DNR$ — marginal increase
- **Head-specific $x,y,z$**: retain original projection, scale element-wise by a learnable data-independent vector of size $R$, so growth is $DPR \to DP + PR$ — additive rather than multiplicative
- MIMO models are **parameter-matched** to SISO by reducing MLP inner dimension

---

## Mamba-3 Architecture


![[Mamba-3 Architecture.png]]

Macro-structure follows **Llama**: alternating Mamba-3 blocks and SwiGLU FFN blocks with pre-normalization.

### Changes from Mamba-2

| Component | Mamba-2 | Mamba-3 |
|-----------|---------|---------|
| Recurrence | Exponential-Euler SSD | Exponential-Trapezoidal SSD |
| Dynamics | Real scalar $A$ | Complex $A + i\Theta$ (RoPE trick for $\Theta$) |
| Short conv | Explicit, standalone | **Removed** (absorbed into discretization + biases) |
| Post-gate RMSNorm | Required for stability | **Removed** (BCNorm suffices) |
| QK Normalization | Optional | **BCNorm** (RMSNorm on $B,C$ after projection) |
| $B,C$ biases | None | Learnable head-specific channel-wise biases |

### Key Architectural Details

**Updated SSM recurrence**: The SSD layer uses complex-valued exponential-trapezoidal recurrence. The real component $A$ goes through standard SSD; the imaginary component $\Theta$ is handled via the RoPE trick. MIMO can be optionally enabled for stronger decoding.

**BCNorm (QKNorm)**: RMSNorm applied immediately after $B$ and $C$ projections. Stabilizes large-scale training, replacing the post-gate RMSNorm needed in Mamba-2.

**Learnable $B/C$ biases**: Data-independent head-specific biases added after BCNorm. Together with trapezoidal discretization, they introduce sufficient convolution-like behavior to fully replace the explicit short causal convolution.

> 核心架構統整：Mamba-3 將實數與虛數的狀態轉換拆分計算，並運用 RoPE Trick 處理虛數部分 。
> 架構上最顯著的改變是徹底移除了獨立的短卷積層 (short causal convolution) 。
> 同時，引入了 BCNorm 進行特徵正規化以穩定訓練，進而汰換掉 Mamba-2 中容易受限的 post-gate RMSNorm 。
> 最後配合可學習的通道偏差值 (biases)，整體架構變得更為精簡 。
---

## Critique

#### **Is exponential-trapezoidal truly necessary?** 
The second-order error reduction ($O(\Delta_t^3)$ vs $O(\Delta_t^2)$) is theoretically cleaner, but the empirical gains may partly come from the implicit convolution effect rather than integration accuracy per se. 
It is unclear whether a wider explicit convolution (width $> 1$) in Mamba-2 could achieve equivalent results at lower architectural complexity.

#### **MIMO training cost vs. serving gains**: 
MIMO raises decoding arithmetic intensity, which benefits long-running inference services. 
However, it increases training compute by $R\times$. 
The net benefit depends heavily on the inference-to-training compute ratio of the deployment scenario.

#### **Complex dynamics vs. attention for positional reasoning**:
Complex SSMs introduce data-dependent rotations (RoPE-equivalent), which address local periodic patterns. 
Whether rotational latent dynamics can approximate the full relative positional reasoning performed by standard attention — especially for long-range dependencies — remains an open empirical question.
