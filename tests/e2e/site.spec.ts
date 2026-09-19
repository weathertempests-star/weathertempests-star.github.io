import { expect, test } from '@playwright/test';

test('navigation, responsive layout and keyboard skip link', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  for (const path of ['/', '/blog/', '/notes/', '/cv/', '/search/']) {
    const response = await page.goto(path);
    expect(response?.ok()).toBeTruthy();
    await expect(page.locator('h1')).toBeVisible();
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
    ).toBeTruthy();
    await expect(page.getByRole('navigation', { name: '主要導覽' })).toBeVisible();
  }
  await page.goto('/');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: '跳至主要內容' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).toBeFocused();
  expect(errors).toEqual([]);
});

test('tag filtering persists in URL and restores with browser history', async ({ page }) => {
  await page.goto('/blog/');
  const tags = page.locator('.tag-filter[data-tag]:not([data-tag=""])');
  test.skip((await tags.count()) === 0, 'No published tags yet');
  const firstTag = tags.first();
  const tag = await firstTag.getAttribute('data-tag');
  await firstTag.click();
  await expect(firstTag).toHaveAttribute('aria-pressed', 'true');
  expect(new URL(page.url()).searchParams.get('tag')).toBe(tag);
  await page.reload();
  await expect(firstTag).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: '全部', exact: true }).click();
  expect(new URL(page.url()).searchParams.has('tag')).toBeFalsy();
  await page.goBack();
  await expect(firstTag).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.entry-card:visible').first()).toBeVisible();
});

test('Chinese and English full-text search, no results and draft exclusion', async ({ page }) => {
  let articleUrl: string | null = null;
  for (const collection of ['/notes/', '/blog/']) {
    await page.goto(collection);
    const first = page.locator('.entry-card h3 a').first();
    if (await first.count()) {
      articleUrl = await first.getAttribute('href');
      break;
    }
  }
  if (!articleUrl) {
    await page.goto('/search/');
    await expect(page.getByText('還沒有已發布的文章或研究筆記。', { exact: false })).toBeVisible();
    return;
  }
  await page.goto(articleUrl);
  const title = await page.locator('h1').innerText();
  const body = await page.locator('.prose').innerText();
  const terms = [title, body.match(/[A-Za-z]{4,}/)?.[0]].filter((term): term is string =>
    Boolean(term),
  );
  await page.goto('/search/');
  const input = page.locator('.pagefind-ui__search-input');
  await expect(input).toBeVisible();
  for (const term of terms) {
    await input.fill(term);
    await expect(
      page.locator('.pagefind-ui__result-title').filter({ hasText: title }).first(),
    ).toBeVisible();
  }
  await page.locator('.pagefind-ui__result-title a').filter({ hasText: title }).first().click();
  expect(new URL(page.url()).pathname).toBe(articleUrl);
  await page.goto('/search/');
  for (const term of ['qzxnonexistent987', 'PRIVATE_DRAFT_SENTINEL']) {
    await page.locator('.pagefind-ui__search-input').fill(term);
    await expect(page.locator('.pagefind-ui__message')).toContainText('找不到');
    await expect(page.locator('.pagefind-ui__result')).toHaveCount(0);
  }
});

test('math, code highlighting, article anchors, image and print stylesheet', async ({ page }) => {
  for (const collection of ['/notes/', '/blog/']) {
    await page.goto(collection);
    const first = page.locator('.entry-card h3 a').first();
    if (!(await first.count())) continue;
    await first.click();
    const math = page.locator('.katex-display').first();
    if (await math.count()) await expect(math).toBeVisible();
    const code = page.locator('pre.astro-code').first();
    if (await code.count()) await expect(code).toBeVisible();
    const anchor = page.getByRole('navigation', { name: '文章目錄' }).getByRole('link').first();
    if (await anchor.count()) {
      await anchor.click();
      await expect(page).toHaveURL(/#/);
    }
    for (const image of await page.locator('.prose img').all()) {
      await expect(image).toBeVisible();
      expect(
        await image.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0),
      ).toBeTruthy();
    }
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
    ).toBeTruthy();
  }
  await page.goto('/cv/');
  await expect(page.getByRole('button', { name: '列印 / 另存 PDF' })).toBeVisible();
  await page.emulateMedia({ media: 'print' });
  await expect(page.locator('.site-header')).toBeHidden();
  await expect(page.locator('#print-cv')).toBeHidden();
  await expect(page.locator('.cv-content')).toBeVisible();
});

test('unknown routes use the 404 page', async ({ page }) => {
  const response = await page.goto('/not-a-published-page-74832/');
  expect(response?.status()).toBe(404);
  await expect(page.getByRole('heading', { name: '這一頁，還沒寫到。' })).toBeVisible();
  await page.getByRole('link', { name: '回到首頁', exact: true }).click();
  await expect(page).toHaveURL('/');
});

test('content and navigation remain accessible without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('/blog/');
  await expect(page.locator('.tag-filters')).toBeHidden();
  await expect(page.getByRole('heading', { name: '部落格.' })).toBeVisible();
  await page.goto('/search/');
  await expect(page.locator('#search-fallback, .search-panel .empty-state')).toBeVisible();
  await context.close();
});
