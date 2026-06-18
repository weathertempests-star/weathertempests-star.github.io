---
title: "日誌模板化 (Templatize)"
description: "插件式解析器架構，將原始日誌正規化為模板與參數"
tags: [stage-1, parser, templatize]
date: 2025-01-01
---

# 日誌模板化

> **核心目標**：將結構相同但參數不同的日誌歸納為同一模板，降低後續嵌入的語義差異。

[[Preprocessing|← Stage I 總覽]]

---

## 解析範例

```
原始：  CreateFile C:\Windows\System32\cmd.exe SUCCESS
模板：  CreateFile <PATH> SUCCESS
參數：  C:\Windows\System32\cmd.exe
```

正規化後，不同路徑的相同操作會映射至同一模板，提升聚類與異常偵測的穩定性。

---

## 四步驟處理流程

```
Step 1  解析器初始化（importlib 動態載入）
   ↓
Step 2  逐行解析
        ├─ 是 Reg* 操作 → RegistryParser（深層樹）
        └─ 否           → 標準 Parser
   ↓
Step 3  生成 LogID（{檔名}_{行號}）
   ↓
Step 4  批次輸出至 data/Intermediate_data/
```

---

## 輸出欄位

### 解析器啟用時

| 欄位 | 範例 |
|------|------|
| `LogID` | `abc123_0` |
| `Template` | `CreateFile <PATH> SUCCESS` |
| `Parameters` | `C:\Windows\System32\cmd.exe` |
| `OriginalLog` | `CreateFile C:\Windows\... SUCCESS` |

### 解析器停用時

| 欄位 | 範例 |
|------|------|
| `LogID` | `abc123_0` |
| `ConcatenatedLog` | `CreateFile C:\Windows\... SUCCESS` |

---

## 支援的解析器

| 解析器 | 演算法特性 |
|--------|-----------|
| `drain` | 前綴樹分群，速度快，預設選擇 |
| `spell` | 最長公共子序列，適合結構多變日誌 |
| `lenma` | 長度感知，對正則欄位友善 |

---

## 快速使用

```python
from preprocess.preprocess import LogLoader

# 使用預設 Drain
loader = LogLoader()
loader.load_logs(ratio=0.3)

# 自訂解析器與配置
loader = LogLoader(
    parser_name="drain",
    parser_config={
        'standard': {'depth': 5, 'st': 0.6},
        'registry': {'depth': 7, 'st': 0.5}
    }
)
loader.load_logs(columns=["Operation", "Path", "Result"])
```

---

## 相關筆記

- [[Add-Parser|撰寫新解析器]] — 自訂解析器開發指南
- [[Preprocessing|Stage I 預處理總覽]] — 完整流程