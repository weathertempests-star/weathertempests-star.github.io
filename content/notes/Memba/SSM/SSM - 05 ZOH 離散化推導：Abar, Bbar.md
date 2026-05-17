---
type: atomic
tags:
  - Memba
  - model-structure
  - SSM
status: active
created: 2026-05-15
updated: 2026-05-15
authors: WeiTa
---

## 這張卡在整體脈絡扮演什麼角色
**從連續 ODE 推到離散遞迴時，$\bar{A},\bar{B}$ 怎麼精確得到？**

這一步把控制理論的「解析解」變成深度學習裡可實作的遞迴更新式。

---

## Zero-Order Hold (ZOH) 數學推導

> ZOH 假設在時間步長 $\Delta$ 內，輸入訊號 $u(t)$ 保持不變。
> i.e.  $u(t) = u_k$ | $t \in [t_k, t_k + \Delta]$。

**推導步驟：**

1. **精確解形式**：

   對於 $x'(t) = Ax(t) + Bu(t)$，在區間 $[t_k, t_{k+1}]$ 的解為：

   $$x(t_{k+1}) = e^{A(t_{k+1}-t_k)}x(t_k) + \int_{t_k}^{t_{k+1}} e^{A(t_{k+1}-\tau)}Bu(\tau) d\tau$$

2. **代入 $\Delta$ 與恆定輸入 $u_k$**：

   令 $t_{k+1} - t_k = \Delta$，則：

   $$x_{k+1} = e^{A\Delta}x_k + \left( \int_0^{\Delta} e^{A(\Delta-\tau)} d\tau \right) Bu_k$$

3. 變數變換與積分求解 (關鍵推導步)
	為了求解積分項，令 $s = t_{k+1} - \tau$，則 $ds = -d\tau$：
- 當 $\tau = t_k \implies s = \Delta$
    
- 當 $\tau = t_{k+1} \implies s = 0$
	代入積分式：

$$\int_{t_k}^{t_{k+1}} e^{A(t_{k+1}-\tau)} d\tau = \int_{\Delta}^{0} e^{As} (-ds) = \int_{0}^{\Delta} e^{As} ds$$

	執行定積分（假設 $A$ 可逆）：

$$\int_{0}^{\Delta} e^{As} ds = \left[ A^{-1} e^{As} \right]_0^\Delta = A^{-1}(e^{A\Delta} - I)$$
4. **求解離散參數**：
	透過上述推導，我們得到與 $x_{k+1} = \bar{A}x_k + \bar{B}u_k$ 對應的離散矩陣：
   
   - **$\bar{A}$ (離散轉移矩陣)**：

     $$\bar{A} = \exp(\Delta A)$$

   - **$\bar{B}$ (離散輸入矩陣)**：

     令 $s = \Delta - \tau$，則 $ds = -d\tau$：

     $$\bar{B} = \left( \int_0^{\Delta} e^{As} ds \right) B = A^{-1}(\exp^{\Delta A} - I)B$$

> 在 Mamba 的實作中，若考慮數值穩定性或簡化，此項有時會近似為 $\bar{B} \approx (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B$。

---

## 下一步
- 讀 $\Delta$ 的物理意義（與 Mamba 的 input-dependent gating 連結）：[[SSM/SSM - 06 步長 Δ 的意義：時間解析度與動態門控]]
- 讀離散後的遞迴形式如何變成卷積：[[SSM/SSM - 07 遞迴視角 vs 卷積視角：為何可平行訓練]]
