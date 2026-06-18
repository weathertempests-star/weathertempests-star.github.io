---
type: permanent
tags:
  - Transformer
status: active
created: 2026-05-13
updated: 2026-05-13
authors: WeiTa
---
	---

### 1. 計算複雜度：Self-Attention 的 $O(n^2)$ 困境

Transformer 的核心機制是 **Global Attention**，要求序列中的每一個 Token 與其他所有 Token 進行交互。
- **機制**：計算 $QK^T$ 時，Query 矩陣 ($n \times d$) 與 Key 矩陣 ($d \times n$) 相乘，產生一個 $n \times n$ 的 Attention 矩陣。
- **複雜度**：計算成本隨序列長度 $n$ 的平方增加。當序列長度從 1k 增加到 10k，計算量增加 100 倍。
- **公式**：
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
### 2. KV Cache：空間成本與內存占用

在自迴歸（Autoregressive）推理過程中，為了避免重複計算已生成的 Token，系統會緩存所有歷史 Token 的 Key ($K$) 與 Value ($V$) 向量。
- **線性增長**：KV Cache 的存儲需求隨序列長度 $n$ 線性增長。

- **容量壓力**：在長序列（如 100k tokens）下，KV Cache 會佔據數十 GB 的 VRAM，甚至超過模型參數本身的大小，導致memory溢出（OOM）。
- 
- **計算代價**：每生成一個新 Token，都需要讀取整個 KV Cache。

### 3. Memory Bandwidth Pressure：內存頻寬瓶頸

Transformer 的推理過程通常是 **Memory-bound**（受限於內存讀取速度）而非 **Compute-bound**（受限於運算能力）。

- **讀取開銷**：生成每一個新 Token 時，GPU 必須將龐大的 KV Cache 從 HBM（高頻寬顯存）搬運到 SRAM（片上高速緩存）中。

- **頻寬利用**：當序列變長，搬運 KV Cache 的時間遠超過矩陣乘法的運算時間，導致 GPU 算力利用率低下，這是長序列推理速度慢的主因。
    

### 4. Inference Latency：推論延遲

- **逐字生成限制**：Transformer 必須串行生成 Token。隨著 Context 增長，每一步的內存讀取量（KV Cache）增加，導致單個 Token 的生成時間（Per-token Latency）隨長度線性上升。
    
- **長上下文瓶頸**：在處理超長文檔（如整本書籍或代碼庫）時，預填充（Prefill）階段的 $O(n^2)$ 計算與生成階段的內存頻寬壓力會導致顯著的系統卡頓。
    
