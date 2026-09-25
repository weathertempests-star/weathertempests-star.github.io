import { expect, test } from '@playwright/test';

test('research topics filter notes, preserve history and tolerate unknown tags', async ({
  page,
}) => {
  await page.goto('/notes/');
  await expect(page.locator('h1')).toContainText('研究筆記');
  await expect(page.locator('.entry-card')).toHaveCount(10);
  for (const tag of ['Agent 系統', '模型訓練', 'AI 安全', '機密運算']) {
    await page.getByRole('button', { name: tag, exact: true }).click();
    expect(new URL(page.url()).searchParams.get('tag')).toBe(tag);
    const visible = page.locator('.entry-card:visible');
    expect(await visible.count()).toBeGreaterThan(0);
    for (const card of await visible.all()) {
      expect(JSON.parse((await card.getAttribute('data-entry-tags'))!)).toContain(tag);
    }
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
  await page.goto('/notes/?tag=unknown-tag');
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

test('home links to three research notes; paper metadata and mobile contents remain usable', async ({
  page,
  isMobile,
}) => {
  await page.goto('/');
  await expect(page.locator('.home-research .entry-title')).toHaveCount(3);
  await page.locator('.home-research .entry-title').first().click();
  await expect(page.locator('.paper-info')).toBeVisible();
  await expect(page.locator('.paper-title a')).toHaveAttribute(
    'href',
    /^https:\/\/arxiv\.org\/abs\/\d+\.\d+v\d+/,
  );
  await expect(page.locator('.paper-authors')).not.toBeEmpty();
  await expect(page.locator('.paper-status')).toHaveText(/預印本|已出版/);
  const toc = page.locator('.toc-disclosure');
  expect(await toc.evaluate((node: HTMLDetailsElement) => node.open)).toBe(!isMobile);
  if (isMobile) await toc.locator('summary').click();
  await page.getByRole('navigation', { name: '文章目錄' }).getByRole('link').first().click();
  await expect(page).toHaveURL(/#/);
});

test('retired note URLs return 404', async ({ page }) => {
  for (const slug of ['gradient-descent', 'optee-from-zero-01']) {
    const response = await page.goto(`/notes/${slug}/`);
    expect(response?.status()).toBe(404);
    await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content', /noindex/);
  }
});

test('research articles and paper sources remain readable without JavaScript', async ({
  browser,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('/notes/');
  await expect(page.locator('.entry-title')).toHaveCount(10);
  await expect(page.locator('.tag-filters')).toBeHidden();
  await page.locator('.entry-title').first().click();
  await expect(page.locator('.paper-source-link')).toBeVisible();
  await expect(page.getByRole('navigation', { name: '文章目錄' })).toBeVisible();
  await context.close();
});

test('published research diagrams render with accessible explanations', async ({ page }) => {
  await page.goto('/notes/');
  const urls = await page
    .locator('.entry-title')
    .evaluateAll((links) => links.map((link) => link.getAttribute('href')!));
  let diagrams = 0;
  for (const url of urls) {
    await page.goto(url);
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
      const diagramFits = await figure
        .locator('.mermaid-canvas')
        .evaluate((canvas) => canvas.scrollWidth <= canvas.clientWidth + 1);
      expect(diagramFits).toBe(true);
    }
  }
  expect(diagrams).toBeGreaterThan(0);
});
