---
title: '理解梯度下降：從直覺到第一行程式'
description: '沿著函數下降的方向，一步步靠近最小值。用一個簡單的二次函數，串起直覺、數學公式與 Python 實作。'
date: 2026-09-18
tags: ['機器學習', '數學', 'Python']
draft: false
featured: true
sample: true
---

## 問題與直覺

這份示範筆記用來展示數學推導、程式碼與參考文獻的排版，並非原創研究成果。

假設我們想找出函數 $f(x) = (x - 3)^2$ 的最小值。從圖形可以看出，最低點位於 $x = 3$。如果不知道答案，能不能只靠目前位置的資訊，逐步找到它？

**梯度下降（Gradient Descent）** 的想法是：沿著函數值下降的方向，前進一小步，再重新評估。

## 更新公式

對可微分的目標函數 $f$，一次更新寫成：

$$
x_{t+1} = x_t - \eta \, \nabla f(x_t)
$$

其中 $\eta > 0$ 是學習率，決定每次移動的步長；$\nabla f(x_t)$ 則是目前位置的梯度。

### 二次函數的例子

對 $f(x) = (x - 3)^2$ 微分，得到：

$$
\nabla f(x) = 2(x - 3)
$$

取 $x_0 = 0$、$\eta = 0.1$，第一次更新為 $x_1 = 0.6$，比起起點更接近 $3$。

### 學習率的影響

這個特定例子的誤差滿足：

$$
x_{t+1} - 3 = (1 - 2\eta)(x_t - 3)
$$

因此，當 $0 < \eta < 1$ 時，誤差絕對值逐步縮小。這個範圍來自本例的函數，不能直接套用到所有最佳化問題。

## 用 Python 驗證

```python
def gradient_descent(x=0.0, learning_rate=0.1, steps=80):
    history = [x]
    for _ in range(steps):
        gradient = 2 * (x - 3)
        x = x - learning_rate * gradient
        history.append(x)
    return x, history

minimum, trajectory = gradient_descent()
print(f"x ≈ {minimum:.6f}")  # x ≈ 3.000000
```

可以調整 `learning_rate`，觀察收斂、震盪與發散之間的差異。

## 接下來的問題

- 如果函數有多個局部最小值，起點會如何影響結果？
- 當參數變成向量時，如何解讀梯度？
- 隨機梯度下降與完整梯度下降有什麼差別？

## 參考資料

1. Stephen Boyd 與 Lieven Vandenberghe，[_Convex Optimization_](https://web.stanford.edu/~boyd/cvxbook/)，第 9 章。
2. Ian Goodfellow、Yoshua Bengio 與 Aaron Courville，[_Deep Learning_](https://www.deeplearningbook.org/contents/optimization.html)，第 8 章。

延伸閱讀：[如何整理自己的學習筆記](/blog/learning-in-public/)。
