# 未完筆記 · 個人部落格與學術作品集

繁體中文、現代藍白版面的個人網站。使用 Astro 7、TypeScript、Markdown、KaTeX 與 Pagefind，輸出靜態檔案，部署到 GitHub Pages。

包含首頁、部落格、研究筆記、關於與履歷、全文搜尋、404、RSS 與 sitemap。首版個人資料是明確標示的待填內容；兩篇公開文章是排版示範。

## 本機預覽

需要 Node.js 24（亦支援 22.12 以上版本）及 npm。

```bash
npm ci
npm run dev
```

開啟終端機顯示的網址，預設是 `http://localhost:4321`。一般編輯會即時更新。搜尋索引在正式建置時建立，測試搜尋請改用：

```bash
npm run check
npm run build
npm run preview
```

若執行環境禁止寫入使用者設定目錄，可在指令前設定 `ASTRO_TELEMETRY_DISABLED=1`，例如 `ASTRO_TELEMETRY_DISABLED=1 npm run build`。

## 填入你的資料

修改 [`src/data/profile.ts`](src/data/profile.ts)，即可更新站名、姓名、簡介、研究領域、Email、社群連結、學經歷、研究成果與技能。不需要修改頁面版型。

- `siteName`：網站名稱，預設「未完筆記」。
- `name`、`role`、`bio`：顯示名稱、身分簡述與自我介紹。
- `description`：網站預設 SEO 及 RSS 說明。
- `interests`：研究領域名稱陣列。
- `email`、`links`：只填你想公開的資訊；空值不會產生聯絡按鈕。
- `education`、`experience`、`publications`：每筆填入 `period`、`title`、`organization`、`description`，可加上 `url`。
- `skills`：每筆包含 `category` 與 `items` 陣列。

例如把 `education: []` 改為以下格式，並將所有提示替換為真實資料：

```typescript
education: [{
  period: '就讀期間',
  title: '學位／系所',
  organization: '學校名稱',
  description: '研究方向或補充說明',
}],
links: [{ label: 'GitHub', url: 'https://github.com/weathertempests-star' }],
skills: [{ category: '程式語言', items: ['你使用的語言'] }],
```

履歷頁的「列印 / 另存 PDF」使用瀏覽器列印功能。選擇「另存為 PDF」即可下載，列印版會隱藏網站導覽和按鈕。

## 新增文章與筆記

部落格放在 `src/content/blog/`，研究筆記放在 `src/content/notes/`。新增 `.md` 檔案，例如 `my-first-post.md`：

```markdown
---
title: '我的第一篇文章'
description: '用一兩句話描述文章內容。'
date: 2026-09-19
tags: ['閱讀', '學習']
draft: false
featured: true
sample: false
---

## 從一個問題開始

在這裡寫下你的內容。

### 我的理解

使用 Markdown 撰寫文字、清單、連結與程式碼。
```

`title`、`description`、`date` 必填，缺漏或日期無法解析時建置會失敗並指出檔案。建議日期固定寫成 `YYYY-MM-DD`；網站顯示日期使用台北時區。

| 欄位       | 用途                                            | 預設    |
| ---------- | ----------------------------------------------- | ------- |
| `tags`     | 列表篩選與文章標籤                              | `[]`    |
| `draft`    | 不產生公開頁面，不進入列表、RSS、sitemap 或搜尋 | `false` |
| `featured` | 顯示於首頁精選區，最多兩篇，按日期由新到舊      | `false` |
| `sample`   | 顯示「示範內容」標記與文章提示                  | `false` |

檔名決定文章網址，例如 `/blog/my-first-post/`。建議使用英文小寫與連字號，發布後保留檔名以維持網址穩定。修改標題不會改變網址。日期只用於排序，**不提供定時發布**；未準備公開的內容請設定 `draft: true`。草稿在本機開發模式也不會產生頁面；預覽草稿時可暫時改為 `false`，提交前再還原。

若儲存庫為公開，Markdown 原始檔仍可從 GitHub 讀取；`draft` 控制的是網站發布狀態。

### 圖片、公式與引用

將圖片放入 `public/images/`，在 Markdown 使用站內絕對路徑及圖片描述：

```markdown
![描述這張圖片的內容](/images/my-image.png)

行內公式：$E = mc^2$

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

## 參考資料

1. 作者，[文章或論文標題](https://example.com/paper)。
```

程式碼區塊在三個反引號後加上 `python`、`javascript` 等語言名稱即可高亮。二級與三級標題會自動成為文章目錄。首版使用手動參考文獻，未整合 BibTeX。

可直接在 GitHub 網頁新增／修改 Markdown，提交到 `main` 後會自動重新部署。也可以在本機編輯，透過 Git 推送。

## 部署到 GitHub Pages

此版本使用個人網站根網址 `https://<帳號>.github.io/`，不支援 `/repo/` 子路徑。正式部署網址會由 GitHub Actions 的儲存庫資訊自動設定，不必把帳號寫進程式碼。

1. 在 GitHub 建立名為 **`<你的帳號>.github.io`** 的儲存庫。使用 GitHub Free 時選擇 Public。若要從這份資料夾直接推送，建立時不要勾選額外的 README、License 或 `.gitignore`。
2. 在此專案根目錄開啟終端機；若尚未初始化 Git，可使用以下指令。將 `YOUR_USERNAME` 替換為實際帳號。

   ```bash
   git init -b main
   git add .
   git commit -m "Create personal blog and research website"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_USERNAME.github.io.git
   git push -u origin main
   ```

   若已有 Git 儲存庫或 remote，先確認現有設定，再接上正確的遠端。Git 登入請使用 GitHub 提供的驗證方式，不要把密碼或權杖寫進專案。

3. 進入 GitHub 儲存庫的 **Settings → Pages → Build and deployment → Source**，選擇 **GitHub Actions**。
4. 到 **Actions → Check and deploy personal website** 查看執行結果。如果第一次執行在啟用 Pages 前已失敗，設定完成後重新執行工作流程。
5. 成功後開啟 `https://YOUR_USERNAME.github.io/`，確認文章可直接開啟、圖片正常、搜尋能找到中英文內容，以及履歷列印正常。

工作流程先進行型別檢查、建置、輸出驗證與瀏覽器測試，再發布 `dist/`。Pull request 只檢查，不會部署；只有 `main` 的推送或手動執行才會部署。部署使用 GitHub 自動提供的權限，不需要額外設定網站服務金鑰。

如果儲存庫名稱不符合個人網站格式，建置會提示修正，避免產生錯誤的站內連結。若部署失敗，先看 Actions 的失敗步驟；建置失敗不會取代上一個成功部署的版本。

本機需要測試正式網址時，可將 `.env.example` 複製為 `.env`，設定：

```dotenv
SITE_URL=https://YOUR_USERNAME.github.io
```

`.env` 不會提交至 Git。未設定時使用 `http://localhost:4321`；GitHub Actions 一律使用實際儲存庫網址。此設定會套用到 canonical、Open Graph、RSS 與 sitemap。

## 驗證與專案維護

```bash
npm run check
npm run build
npm test
npx playwright install chromium
npm run test:e2e
```

瀏覽器測試包含桌面與手機排版、中文／英文搜尋、標籤網址與上一頁行為、文章目錄、公式、圖片、履歷列印及 404。單元與建置驗證涵蓋部署網址、必要輸出、站內連結、標題錨點及草稿排除。

正式網站的檔案在 `dist/`，不必提交；請提交來源檔、設定與 `package-lock.json`。修改套件後同步更新鎖定檔，讓本機與 GitHub Actions 使用相同版本。

參考：[GitHub Pages 快速入門](https://docs.github.com/en/pages/quickstart)、[Astro 部署文件](https://docs.astro.build/en/guides/deploy/github/)、[Pagefind 中文搜尋](https://pagefind.app/docs/multilingual/)。
