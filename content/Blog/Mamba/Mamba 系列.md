---
type: moc
tags:
  - Memba
  - Language-Model
status: active
created: 2026-05-13
updated: 2026-05-15
authors: WeiTa
---
---
## 0. Transformer的問題是甚麼
- [[Transformer 的問題是甚麼]]

## 1. SSM (State Space Model) 理論地基

用最少必要的數學語言建立 SSM 的基本問題：
1.  為什麼需要記憶
2. 狀態怎麼表示
3. 如何從連續時間走到可訓練的離散形式。

- [[SSM (State Space Model)|SSM 索引（線性脈絡 → 卡片式筆記detail)]]：整體地圖，確認 SSM 在解決什麼問題，再依需求往下拆到各個子筆記。
- 概念
	- [[SSM/SSM - 01 問題與直覺：有限記憶 vs 無限序列|動機與直覺:]]， SSM 為什麼被提出來 - 補足 Transformer 或一般序列模型不容易處理的**長期記憶**問題。
	- [[SSM/SSM - 02 連續時間 LTI SSM 定義 (A,B,C,D)|連續時間定義與符號：]]，認識 $A,B,C,D$ 分別代表什麼，避免後面看到公式卻不知道每個符號在扮演哪個角色。
	- [[SSM/SSM - 03 參數矩陣 A-B-C-D 的幾何與物理解釋|參數與矩陣扮演的角色：]]為什麼 $A$ 會決定記憶如何衰減、保留或轉換。
- 實作  
	- [[SSM/SSM - 05 ZOH 離散化推導：Abar, Bbar|連續 → 離散（ZOH / $\Delta$）：]]，說明理論模型怎麼變成實際可計算的**離散**版本。
	- [[SSM/SSM - 08 卷積核 K 的構造與 FFT 複雜度| 遞迴 → 卷積（FFT）：]]為什麼同一個系統可以改寫成卷積形式，以及這樣做如何影響計算效率。
	- [[SSM/SSM - 11 S4：Structured State Space 與對角化動機|HiPPO 與 S4（結構化 $A$）：]]，更進一步的結構化設計，讓模型既保留表達力，也更容易高效計算。
    

## 2. Mamba 架構演化脈絡

這一段沿著 Mamba 的版本演化來讀：先看 Mamba-1 如何把 SSM 變成內容感知的選擇性模型，再看 Mamba-2 如何把計算改造成更吃硬體，最後看 Mamba-3 如何再把推論效率往前推。第一次閱讀時，可以把每一代都當成對前一代限制的修補。

- [[Mamba/Mamba - 00 演化索引（線性脈絡）|Mamba 演化索引（線性脈絡 → 原子卡深讀）]]：先讀主線，確認每一代是在修正哪個瓶頸，再按代際深入細節。
  - Mamba-1（Selective SSM / $B_t,C_t,\Delta_t$）：
    - [[Mamba/Mamba - 03 Mamba-1：Selective SSM（B_t,C_t,Δ_t）]]，先看 selective SSM 如何讓參數跟輸入內容動態對齊，讓模型不再是固定規則的 SSM。
  - Mamba-1 工程代價與補救（FFT → Scan）：
    - [[Mamba/Mamba - 05 Mamba-1：為何失去 FFT？以及 Parallel Scan 的位置]]，再看這種內容感知帶來了什麼代價，以及為什麼後來要改用 parallel scan 承接計算。
  - Mamba-2（SSD / GEMM 吃滿）：
    - [[Mamba/Mamba - 06 Mamba-2：SSD 的問題設定與限制]]，接著看 Mamba-2 想解決的核心問題是什麼，以及它在哪些情境下特別有價值。
    - [[Mamba/Mamba - 07 Mamba-2：SSD 的矩陣形式與 Block-Chunk 策略]]，這一節把 SSD 改寫成更適合矩陣運算與硬體吞吐的形式，重點是理解為何它能更有效率地跑起來。
  - Mamba-3（Inference-oriented）：
    - [[Mamba/Mamba - 08 Mamba-3：Exponential-Trapezoidal Discretization]]，先看它怎麼把離散化做得更適合推論場景，讓更新規則更穩定也更高效。
    - [[Mamba/Mamba - 09 Mamba-3：Complex-valued Dynamics via RoPE（直覺）]]，再看複數動態與 RoPE 直覺如何幫助模型表達位置與狀態變化。
    - [[Mamba/Mamba - 10 Mamba-3：MIMO 與 memory-bound 觀點]]，最後理解 Mamba-3 為什麼會從計算公式一路走到記憶體瓶頸的硬體觀點。
  - 拓樸演化（組件移除/融合）：
    - [[Mamba 架構的拓樸演化：從 H3 到極致精簡的 Mamba-3]]，這條線不是在講單一公式，而是在看整個架構如何逐步刪減、融合與重組。
        - 深讀：[[Mamba/Mamba - 11 拓樸演化：H3 → Mamba-3（同質化-組件融合）]]，如果想理解每個模組為何被保留或移除，可以再進一步讀這一篇。
    

## 3. Selective Scan 與 Hardware-Aware 設計

- **[] Selective SSM 參數動態生成**：輸入矩陣映射為 $B_t, C_t, \Delta_t$ 的線性投影維度與張量變化。
    
- **[] I/O-Aware 記憶體層級架構**：GPU HBM (High Bandwidth Memory) 與 SRAM 的讀寫成本對比與瓶頸。
    
- **[] Kernel Fusion 實作**：避免中間狀態存取，參數載入 SRAM、局部運算與結果寫回的底層 C/C++/CUDA 邏輯。
    
- **[] Work-efficient Parallel Scan**：前綴和 (Prefix sum) 硬體加速演算法在時序遞迴計算中的並行化策略。
    

## 4. Mamba-3 跨架構對比

- **[] Mamba-3 vs. Transformer**：時間複雜度 ($O(N)$ vs $O(N^2)$)、KV Cache 記憶體消耗及上下文檢索 (In-context learning) 能力對比。
    
- **[] Mamba-3 vs. RWKV**：動態權重與時間衰減 (Time-decay) 機制對比，線性 RNN 的遺忘門控設計差異。
    
- **[] Mamba-3 vs. Hyena**：數據依賴動態卷積核與全局數據獨立卷積核的表示能力邊界。
    
- **[] Mamba-3 vs. RetNet**：Retention 機制的多尺度衰減與分塊遞迴 (Chunkwise Recurrent) 狀態傳遞差異。
    
- **[] Mamba-3 vs. Linear Attention**：核技巧 (Kernel Trick) 展開與結構化矩陣乘法在硬體吞吐量上的上限評估。
    

## 5. SLM 應用與後續架構研究

- **[] SLM 推論效率數據**：低參數規模 (<10B) 下的 Throughput、Memory Footprint 與 Latency 實測指標基準。
    
- **[] 混合架構演化**：MoE-Mamba, Jamba 等 SSM-Transformer 混合架構在特定層次融合的權衡分析。
    
- **[] 長文本 (Long-context) 處理能力**：在 1M+ tokens 任務上的外推性、大海撈針 (NIAH) 測試表現與信息衰減極限。
    
- **[] 模型編輯與知識注入**：MEND / MEMIT 等 Model Editing 技術在非注意力架構上的梯度分解或參數定位適配性。