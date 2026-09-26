# Arm TrustZone-A 與 OP-TEE：從零開發到安全研究

這份藍圖規劃一條從概念、可重現實作到防禦性安全研究的學習路線。  
讀者不需要 TEE 經驗；Linux 與 C 的必要基礎會在用到時補上。  
主線先使用 QEMU 的 Armv8-A 虛擬平台，讓每一步可重現，再將已驗證的實驗移植到 Jetson。  
先練習安全驗證與測試，再透過公開漏洞學習根因與修補分析。

2026-09-26 更新：首篇保留於系列筆記，透過[系列目錄](https://weathertempests-star.github.io/notes/series/optee-from-zero/)依篇次閱讀。  
以下後續篇章仍屬規劃，不代表已發布、實作或測試，也不承諾完成時間。  
環境篇會在實測時才固定軟體版本；程式範例先採可取得的固定版本官方例子，後續自建程式會完整交付原始碼、建置方式與驗收證據。

## 共通完成標準

- 每篇先列出前置知識與操作環境，並解釋新名詞在當章為何必要。
- 可操作章節提供可取得的程式、明確產物、預期輸出與停止條件；失敗時先保留日誌再排除問題。
- 安全結論區分程式行為、平台設定與尚未驗證的假設，不把功能成功當成全面安全保證。
- 每一份自建 CA、TA、測試與量測工具都放在公開版本控制中；讀者能由固定 revision 重建。

## 章節藍圖

### 01｜TrustZone、OP-TEE 與信任邊界

- **狀態：**首篇內容已完成；不需先備知識。  
- **操作環境與程式：**概念與案例分析，不安裝工具或執行程式。  
- **可交付成果：**以元件圖與資料表說明 Normal World、Secure World、資產與不可信輸入。  
- **驗收條件：**文章說明 Linux root 為何不等同 Secure World，並指出「金鑰不匯出」仍需要呼叫者授權。  
- **第一手參考：**[OP-TEE About](https://optee.readthedocs.io/en/latest/general/about.html)、[OP-TEE Core](https://optee.readthedocs.io/en/latest/architecture/core.html)。

### 02｜Linux 工具、QEMU 啟動與 xtest

- **狀態：**規劃中；先備為第 01 篇，終端機操作於本篇補充。  
- **操作環境與程式：**以 Ubuntu x86_64 主機、QEMU Armv8-A 與官方 OP-TEE QEMU 建置流程為起點；版本在實測時固定。  
- **可交付成果：**啟動紀錄、`xtest` 結果與環境 manifest。  
- **驗收條件：**讀者可由乾淨工作目錄啟動映像，並區分 QEMU 啟動失敗、Linux 問題與 TEE runtime 問題。  
- **第一手參考：**[QEMU v8 build](https://optee.readthedocs.io/en/latest/building/devices/qemu.html#qemu-v8)、[Linux OP-TEE driver](https://docs.kernel.org/tee/op-tee.html)。

### 03｜最低 C 基礎及首個 CA／TA

- **狀態：**規劃中；先備為第 02 篇。  
- **操作環境與程式：**第 02 篇的 QEMU 環境與固定 revision 的官方範例；補充 pointer、array、return value、Makefile。  
- **可交付成果：**可建置的最小 CA／TA、UUID 與一次成功呼叫紀錄。  
- **驗收條件：**能指出 CA 使用 Client API、TA 使用 Internal API，且 UUID、command ID 與參數協定可對應到原始碼。  
- **第一手參考：**[GlobalPlatform API in OP-TEE](https://optee.readthedocs.io/en/latest/architecture/globalplatform_api.html)、[Trusted Applications](https://optee.readthedocs.io/en/latest/architecture/trusted_applications.html)。

### 04｜參數、buffer、共享記憶體及驗證

- **狀態：**規劃中；先備為第 03 篇。  
- **操作環境與程式：**QEMU 與第 03 篇程式的固定 revision，新增最小輸入、過長輸入與短輸出測試。  
- **可交付成果：**參數表、CA／TA 兩端的長度與型別驗證，以及負向測試輸出。  
- **驗收條件：**TA 不只信任 CA；測試能確認錯誤來源、輸出長度與 session 是否仍可用。  
- **第一手參考：**[GlobalPlatform API in OP-TEE](https://optee.readthedocs.io/en/latest/architecture/globalplatform_api.html)、[Linux OP-TEE driver](https://docs.kernel.org/tee/op-tee.html)。

### 05｜呼叫流程、日誌與 GDB

- **狀態：**規劃中；先備為第 04 篇。  
- **操作環境與程式：**QEMU、CA／TA 範例與可重現的除錯設定。  
- **可交付成果：**從 context 到 session、command、close 的流程圖，以及一份最小除錯紀錄。  
- **驗收條件：**能分辨 CA log、Linux driver log 與 Secure World log，並用 breakpoint 驗證一條 TA 呼叫路徑。  
- **第一手參考：**[OP-TEE Core](https://optee.readthedocs.io/en/latest/architecture/core.html)、[Trusted Applications](https://optee.readthedocs.io/en/latest/architecture/trusted_applications.html)。

### 06｜RSA-OAEP 與金鑰生命週期

- **狀態：**規劃中；先備為第 04、05 篇。  
- **操作環境與程式：**QEMU 與完整交付的教學 TA；使用明確的 OAEP 參數與測試向量。  
- **可交付成果：**金鑰建立、使用、釋放的程式與正常、邊界、互通測試。  
- **驗收條件：**不匯出私鑰；可區分同一 session 與重開 session 的金鑰生命週期，也不把 round-trip 當成唯一正確性證據。  
- **第一手參考：**[RFC 8017 RSAES-OAEP](https://www.rfc-editor.org/rfc/rfc8017.html#section-7.1)、[GlobalPlatform API in OP-TEE](https://optee.readthedocs.io/en/latest/architecture/globalplatform_api.html)。

### 07｜簽章服務與呼叫者授權

- **狀態：**規劃中；先備為第 06 篇。  
- **操作環境與程式：**QEMU 與完整交付的簽章服務範例。  
- **可交付成果：**只回傳 signature 的 TA、呼叫者政策、拒絕與允許測試。  
- **驗收條件：**說明 root-only 裝置權限的限制，並以測試證明未授權請求不會借用私鑰。  
- **第一手參考：**[Trusted Applications](https://optee.readthedocs.io/en/latest/architecture/trusted_applications.html)、[GlobalPlatform API in OP-TEE](https://optee.readthedocs.io/en/latest/architecture/globalplatform_api.html)。

### 08｜安全儲存與 rollback

- **狀態：**規劃中；先備為第 07 篇。  
- **操作環境與程式：**QEMU 與具明確 backend 說明的 persistent-object 範例。  
- **可交付成果：**持久化物件、刪除與重啟後讀取測試，以及 storage／rollback 假設表。  
- **驗收條件：**不把 API 呼叫當作 RPMB 或 rollback 防護的證明；能指出平台 backend 與保護範圍。  
- **第一手參考：**[OP-TEE Secure storage](https://optee.readthedocs.io/en/latest/architecture/secure_storage.html)。

### 09｜安全測試與 fuzzing

- **狀態：**規劃中；先備為第 04、08 篇。  
- **操作環境與程式：**QEMU、已完成的 CA／TA 與受控測試 harness。  
- **可交付成果：**輸入分類、固定 regression suite、fuzzing 範圍與異常的最小重現方法；用自行設計的錯誤案例驗證測試工具。  
- **驗收條件：**能將 protocol error、TA panic、通訊錯誤分開報告，保存觸發輸入與版本，並確認修正後原案例通過回歸測試。  
- **第一手參考：**[Trusted Applications](https://optee.readthedocs.io/en/latest/architecture/trusted_applications.html)、[OP-TEE Core](https://optee.readthedocs.io/en/latest/architecture/core.html)。

### 10｜公開漏洞與修補分析

- **狀態：**規劃中；先備為第 09 篇。  
- **操作環境與程式：**只選已公開、具修補資訊且可在隔離教學環境分析的案例。  
- **可交付成果：**版本界線、根因、修補 diff、回歸測試與影響範圍說明。  
- **驗收條件：**公告、受影響版本與修補 commit 能互相核對；使用相同回歸案例比較修補前後，清楚區分實測、原始碼推論及尚未重現的部分。  
- **第一手參考：**[OP-TEE 官方安全公告](https://github.com/OP-TEE/optee_os/security/advisories)；選定案例後補上對應公告、修補 commit 與回歸測試。

### 11｜session、keygen 與 invoke 效能

- **狀態：**規劃中；先備為第 06 篇。  
- **操作環境與程式：**QEMU 與完整交付的量測工具；平台、CPU 設定與 warm-up 條件隨結果記錄。  
- **可交付成果：**區分 context、open/keygen、invoke、close 的原始資料與延遲分布。  
- **驗收條件：**不把 key generation 或模擬器成本誤稱為純 world-switch latency，並公開量測限制。  
- **第一手參考：**[OP-TEE Core](https://optee.readthedocs.io/en/latest/architecture/core.html)、[Linux OP-TEE driver](https://docs.kernel.org/tee/op-tee.html)。

### 12｜Jetson 移植與實驗

- **狀態：**規劃中；先備為第 02 至 11 篇中的對應實驗。  
- **操作環境與程式：**實際取得、可安全回復的 Jetson 平台；BSP、OP-TEE 與工具版本會在實測時固定。  
- **可交付成果：**平台 manifest、移植差異表、部署程序、QEMU 與板上結果對照。  
- **驗收條件：**先備份與確認 TA 驗證政策；結論僅涵蓋實測配置，不外推為所有 Jetson 或所有 TrustZone-A 平台。  
- **第一手參考：**[NVIDIA Jetson 文件版本入口](https://docs.nvidia.com/jetson/)、[R35.4.1 OP-TEE 說明](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/Security/OpTee.html)；後者是平台整合的參考，實際操作須改用與板上 BSP 相符的文件。

## 從第 01 篇走到第一個實作

第 01 篇以金鑰服務整理信任邊界；第 02 篇接續記錄 QEMU 啟動與 `xtest` 的實作。  
在此之前不需要 Jetson、私有板卡路徑或既有 TA 專案。  
後續章節沿用已驗證的環境，另行固定範例程式的 revision。  
若需要升級基礎環境，會記錄版本差異並重跑受影響的實驗，讓讀者能從上一章接續。
