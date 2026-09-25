import { expect, test } from '@playwright/test';

test('navigation, metadata, responsive layout and keyboard skip link', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  for (const path of ['/', '/blog/', '/notes/', '/cv/', '/search/']) {
    const response = await page.goto(path);
    expect(response?.ok()).toBeTruthy();
    await expect(page.locator('main h1')).toHaveCount(1);
    await expect(page.locator('main h1')).toBeVisible();
    await expect(page).toHaveTitle(/\S/);
    for (const selector of [
      'meta[name="description"]',
      'meta[property="og:title"]',
      'meta[property="og:description"]',
    ]) {
      await expect(page.locator(selector)).toHaveAttribute('content', /\S/);
    }
    const canonical = await page.locator('link[rel="canonical"]').getAttribute('href');
    expect(canonical).toMatch(/^https?:\/\//);
    expect(new URL(canonical!).pathname).toBe(path);
    await expect(page.locator('meta[property="og:url"]')).toHaveAttribute('content', canonical!);
    await expect(page.locator('main')).not.toContainText(/待填|示範內容/);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
    ).toBeTruthy();
    const navigation = page.getByRole('navigation', { name: '主要導覽' });
    await expect(navigation).toBeVisible();
    for (const [label, href] of [
      ['文章', '/blog/'],
      ['研究筆記', '/notes/'],
      ['關於', '/cv/'],
    ]) {
      await expect(navigation.getByRole('link', { name: label, exact: true })).toHaveAttribute(
        'href',
        href,
      );
    }
  }
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content', /noindex/);
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
    const first = page.locator('.entry-card .entry-title').first();
    if (await first.count()) {
      articleUrl = await first.getAttribute('href');
      break;
    }
  }
  if (!articleUrl) {
    await page.goto('/search/');
    await expect(page.locator('.search-panel .empty-state')).toContainText('還沒有已發布');
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
  const articleUrls = new Set<string>();
  for (const collection of ['/notes/', '/blog/']) {
    await page.goto(collection);
    for (const link of await page.locator('.entry-card .entry-title').all()) {
      const href = await link.getAttribute('href');
      if (href) articleUrls.add(href);
    }
  }
  for (const articleUrl of articleUrls) {
    await page.goto(articleUrl);
    await expect(page.locator('article[data-pagefind-body] h1')).toBeVisible();
    await expect(page.locator('.sample-label, .sample-notice')).toHaveCount(0);
    await expect(page.locator('meta[property="og:type"]')).toHaveAttribute('content', 'article');
    await expect(page.locator('meta[property="article:published_time"]')).toHaveAttribute(
      'content',
      /^\d{4}-\d{2}-\d{2}T/,
    );
    const math = page.locator('.katex-display').first();
    if (await math.count()) await expect(math).toBeVisible();
    const code = page.locator('pre.astro-code').first();
    if (await code.count()) await expect(code).toBeVisible();
    const toc = page.locator('.toc-disclosure');
    if ((await toc.count()) && !(await toc.evaluate((node: HTMLDetailsElement) => node.open))) {
      await toc.locator('summary').click();
    }
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
  await expect(page.locator('main h1')).toBeVisible();
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content', /noindex/);
  await page.getByRole('link', { name: '回到首頁', exact: true }).click();
  await expect(page).toHaveURL('/');
});

test('content and navigation remain accessible without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('/blog/');
  await expect(page.locator('.tag-filters')).toBeHidden();
  await expect(page.locator('main h1')).toHaveCount(1);
  await expect(page.locator('main h1')).toBeVisible();
  await expect(page.locator('.entry-card .entry-title').first()).toBeVisible();
  await page.goto('/search/');
  await expect(page.locator('#search-fallback, .search-panel .empty-state')).toBeVisible();
  await context.close();
});
