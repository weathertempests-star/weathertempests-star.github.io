import { expect, test } from '@playwright/test';

const papers = [
  'actobs',
  'blackwell-confidential-computing',
  'chronicle',
  'collapse',
  'evirca',
  'farsight',
  'jaz',
  'stellar-colosseum',
  'structured-but-fragile',
  'tee-attestation-reproducibility',
];

test('blog topics filter ten papers, preserve history, and restore secondary tags', async ({
  page,
}) => {
  await page.goto('/blog/');
  await expect(page.locator('h1')).toContainText('文章');
  await expect(page.locator('.entry-card')).toHaveCount(10);
  for (const tag of ['Agent 系統', '模型訓練', 'AI 安全', '機密運算']) {
    await page.getByRole('button', { name: tag, exact: true }).click();
    expect(new URL(page.url()).searchParams.get('tag')).toBe(tag);
    const visible = page.locator('.entry-card:visible');
    expect(await visible.count()).toBeGreaterThan(0);
    for (const card of await visible.all())
      expect(JSON.parse((await card.getAttribute('data-entry-tags'))!)).toContain(tag);
    await expect(page.locator('#result-count')).toHaveText(String(await visible.count()));
  }
  await page.reload();
  await expect(page.getByRole('button', { name: '機密運算', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await page.getByRole('button', { name: '全部', exact: true }).click();
  await expect(page.locator('.entry-card:visible')).toHaveCount(10);
  await page.goBack();
  await expect(page.getByRole('button', { name: '機密運算', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await page.goto('/blog/?tag=unknown-tag');
  await expect(page.locator('.entry-card:visible')).toHaveCount(10);
  await expect(page.locator('.secondary-tags')).toBeHidden();
  await page.locator('.more-tags summary').click();
  await page.getByRole('button', { name: '根因分析', exact: true }).click();
  await expect(page.locator('.entry-card:visible')).toHaveCount(1);
  await page.reload();
  await expect(page.getByRole('button', { name: '根因分析', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.locator('.secondary-tags')).toBeVisible();
});

test('notes present one OP-TEE series with its published chapter navigation', async ({ page }) => {
  await page.goto('/notes/');
  await expect(page.locator('h1')).toContainText('系列筆記');
  await expect(page.locator('.series-card')).toHaveCount(1);
  await page.locator('.series-card h2 a').click();
  await expect(page).toHaveURL('/notes/series/optee-from-zero/');
  await expect(page.locator('h1')).toContainText('OP-TEE 從零開始');
  await expect(page.locator('.chapter-list a')).toHaveCount(1);
  await page.locator('.chapter-list a').click();
  await expect(page).toHaveURL('/notes/optee-from-zero-01/');
  await expect(page.locator('.series-navigation .series-name')).toHaveAttribute(
    'href',
    '/notes/series/optee-from-zero/',
  );
  await expect(page.locator('.series-navigation .series-name')).toHaveText('OP-TEE 從零開始');
  await page.locator('.series-navigation summary').click();
  await expect(page.locator('.series-navigation [aria-current="page"]')).toBeVisible();
  await expect(page.locator('.chapter-pagination')).toHaveCount(0);
});

test('home links to articles and the OP-TEE series; paper metadata remains usable', async ({
  page,
  isMobile,
}) => {
  await page.goto('/');
  await expect(page.locator('.reading-paths a[href="/blog/"]')).toBeVisible();
  await expect(page.locator('.reading-paths a[href="/notes/"]')).toBeVisible();
  await expect(page.locator('.home-articles .entry-title')).toHaveCount(3);
  await expect(page.locator('.home-series .series-card')).toHaveCount(1);
  await page.locator('.home-articles .entry-title').first().click();
  await expect(page.locator('.paper-info')).toBeVisible();
  await expect(page.locator('.paper-title a')).toHaveAttribute(
    'href',
    /^https:\/\/arxiv\.org\/abs\/\d+\.\d+v\d+/,
  );
  await expect(page.locator('.paper-authors')).not.toBeEmpty();
  const toc = page.locator('.toc-disclosure');
  expect(await toc.evaluate((node: HTMLDetailsElement) => node.open)).toBe(!isMobile);
  if (isMobile) await toc.locator('summary').click();
  await page.getByRole('navigation', { name: '文章目錄' }).getByRole('link').first().click();
  await expect(page).toHaveURL(/#/);
});

test('legacy paper-note URLs redirect safely while retired URLs are 404', async ({ page }) => {
  const legacy = await page.request.get('/notes/evirca/');
  expect(legacy.ok()).toBeTruthy();
  const legacyHtml = await legacy.text();
  expect(legacyHtml).toMatch(/<meta name="robots" content="noindex, follow"/);
  expect(legacyHtml).toMatch(/rel="canonical" href="[^\"]*\/blog\/evirca\/"/);
  expect(legacyHtml).toMatch(/<a id="destination" href="\/blog\/evirca\/"/);
  expect(legacyHtml).not.toMatch(/data-pagefind-body/);
  const response = await page.goto('/notes/evirca/?from=old#evidence');
  expect(response?.ok()).toBeTruthy();
  await expect(page).toHaveURL(/\/blog\/evirca\/\?from=old#evidence$/);
  await expect(page.locator('.paper-info')).toBeVisible();
  for (const path of [
    '/blog/learning-in-public/',
    '/blog/checking-ai-sources/',
    '/notes/gradient-descent/',
  ]) {
    const missing = await page.goto(path);
    expect(missing?.status()).toBe(404);
    await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content', /noindex/);
  }
});

test('articles, series navigation, and legacy redirects remain readable without JavaScript', async ({
  browser,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('/blog/');
  await expect(page.locator('.entry-title')).toHaveCount(10);
  await expect(page.locator('.tag-filters')).toBeHidden();
  await page.locator('.entry-title').first().click();
  await expect(page.locator('.paper-source-link')).toBeVisible();
  await page.goto('/notes/');
  await page.locator('.series-card h2 a').click();
  await expect(page.locator('.chapter-list a')).toHaveCount(1);
  await page.goto('/notes/evirca/');
  await expect(page).toHaveURL('/blog/evirca/');
  await expect(page.locator('.paper-source-link')).toBeVisible();
  await context.close();
});

test('published paper diagrams render with accessible explanations', async ({ page }) => {
  let diagrams = 0;
  for (const slug of papers) {
    await page.goto(`/blog/${slug}/`);
    const figures = page.locator('.mermaid-figure');
    const count = await figures.count();
    if (!count) continue;
    diagrams += count;
    await expect(page.locator('.mermaid-canvas svg')).toHaveCount(count);
    await expect(page.locator('.mermaid-error:visible')).toHaveCount(0);
    for (const figure of await figures.all()) {
      await expect(figure.locator('svg title')).toHaveText(
        await figure.locator('figcaption').innerText(),
      );
      await expect(figure.locator('.mermaid-description')).not.toBeEmpty();
      expect(
        await figure
          .locator('.mermaid-canvas')
          .evaluate((canvas) => canvas.scrollWidth <= canvas.clientWidth + 1),
      ).toBe(true);
    }
  }
  expect(diagrams).toBeGreaterThan(0);
  await page.goto('/notes/optee-from-zero-01/');
  await expect(page.locator('.mermaid-figure')).toHaveCount(3);
  await expect(page.locator('.mermaid-canvas svg')).toHaveCount(3);
  for (const canvas of await page.locator('.mermaid-canvas').all()) {
    expect(await canvas.evaluate((node) => node.scrollWidth <= node.clientWidth + 1)).toBe(true);
  }
});

test('article tags and related reading lead back into the chosen topic', async ({ page }) => {
  await page.goto('/blog/evirca/');
  await page.locator('.article-tags a').filter({ hasText: '根因分析' }).click();
  await expect(page).toHaveURL(/\/blog\/\?tag=/);
  await expect(page.locator('.entry-card:visible')).toHaveCount(1);
  await expect(page.getByRole('button', { name: '根因分析', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await page.goBack();
  const related = page.locator('.related-articles .entry-title').first();
  const target = await related.getAttribute('href');
  expect(target).not.toBe('/blog/evirca/');
  await related.click();
  await expect(page).toHaveURL(target!);
  await expect(page.locator('.paper-info')).toBeVisible();
  await page.locator('.article-end a').click();
  expect(new URL(page.url()).searchParams.get('tag')).toBe('Agent 系統');
  expect(await page.locator('.entry-card:visible').count()).toBeGreaterThan(1);
});
