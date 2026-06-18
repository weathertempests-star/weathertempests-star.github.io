---
title: "撰寫新解析器"
description: "三步驟插件式解析器開發指南，含 API 規格與完整範例"
tags: [stage-1, parser, dev-guide]
date: 2025-01-01
---

# 撰寫新解析器

> **三步驟完成**：建立檔案 → 實作方法 → 載入使用。

[[Templatize|← 日誌模板化]]

---

## Step 1：建立檔案

在 `Logs Labeling/preprocess/` 目錄下建立 `<parser_name>.py`。

> [!tip] 命名規則
> 檔名必須為**小寫**，例如 `spell.py`、`lenma.py`。

---

## Step 2：實作必要方法

```python
# lenma.py
import re
import pandas as pd
from typing import List, Tuple, Dict

class Parser:
    """標準解析器，系統優先搜尋 DrainParser，備選 Parser"""

    def __init__(self, threshold: float = 0.5, **kwargs):
        self.threshold = threshold
        self.templates = {}

    @staticmethod
    def is_registry_operation(operation: str) -> bool:
        """判斷是否為 Reg* 操作"""
        return operation.startswith('Reg') if operation else False

    def parse(self, log_message: str) -> Tuple[str, List[str]]:
        """解析單行日誌，返回 (模板, 參數列表)"""
        params = []
        template = re.sub(
            r'[A-Z]:\\[\w\\.-]+',
            lambda m: (params.append(m.group()), '<PATH>')[1],
            log_message
        )
        return template, params

    def parse_log_row(self, row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]:
        """解析 DataFrame 行，返回 (模板, 參數, 原始日誌)"""
        log_parts = [str(row[c]).strip() for c in columns
                     if c in row.index and pd.notna(row[c])]
        log_message = " ".join(log_parts)
        template, params = self.parse(log_message)
        return template, params, log_message

    def get_clusters(self) -> List:
        """返回所有已發現的模板"""
        return list(self.templates.keys())
```

---

## 必要方法一覽

| 方法 | 簽章 | 說明 |
|------|------|------|
| `__init__` | `(**kwargs)` | 接收任意配置參數 |
| `parse` | `(str) → (str, List[str])` | 解析單行日誌 |
| `parse_log_row` | `(Series, List[str]) → (str, List[str], str)` | 解析 DataFrame 行 |
| `get_clusters` | `() → List` | 返回模板列表 |
| `is_registry_operation` | `(str) → bool` | 靜態方法，判斷 Reg* |

---

## 可選：RegistryParser（處理 Reg* 事件）

```python
class RegistryParser(Parser):
    def parse_from_row(self, row_dict: Dict) -> Tuple[str, List[str]]:
        """從字典解析，保留完整結構標籤"""
        op = row_dict.get('Operation', '')
        path = row_dict.get('Path', '')
        result = row_dict.get('Result', '')
        template = f"<OP:{op}> <PATH> <RESULT:{result}>"
        return template, [op, path, result]
```

---

## Step 3：載入使用

```python
from preprocess.preprocess import LogLoader

loader = LogLoader(
    parser_name="lenma",          # 對應 lenma.py
    parser_config={
        'standard': {'threshold': 0.7}
    }
)
loader.load_logs(num=50)
```

---

## 類別名稱搜尋順序

```
標準解析器：DrainParser → Parser
註冊表解析器（可選）：RegistryDrainParser → RegistryParser
```

---

## 相關筆記

- [[Templatize|日誌模板化]] — 解析器在流程中的位置
- [[Preprocessing|Stage I 預處理總覽]]