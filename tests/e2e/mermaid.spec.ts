import { expect, test } from '@playwright/test';

const article = '/notes/optee-from-zero-01/';

test('Mermaid diagrams render with accessible labels, readable sizing and print support', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(article);
  const figures = page.locator('.mermaid-figure');
  await expect(figures).toHaveCount(3);
  await expect(page.locator('.mermaid-canvas svg')).toHaveCount(3);
  for (const figure of await figures.all()) {
    const svg = figure.locator('svg');
    await expect(svg).toBeVisible();
    await expect(svg.locator('title')).toHaveText(await figure.locator('figcaption').innerText());
    await expect(svg.locator('desc')).toHaveText(
      await figure.locator('.mermaid-description').innerText(),
    );
    await expect(svg).toHaveAttribute('aria-labelledby', /\S/);
    await expect(svg).toHaveAttribute('aria-describedby', /\S/);
    await expect(figure.locator('.mermaid-error')).toBeHidden();
    await figure.locator('summary').click();
    await expect(figure.locator('pre')).toBeVisible();
    await figure.locator('summary').click();
  }
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
  ).toBeTruthy();
  await page.emulateMedia({ media: 'print' });
  await expect(figures.first().locator('svg')).toBeVisible();
  await expect(figures.first().locator('.mermaid-source')).toBeHidden();
  expect(errors).toEqual([]);
});

test('diagram descriptions and source remain available without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(article);
  for (const figure of await page.locator('.mermaid-figure').all()) {
    await expect(figure.locator('figcaption')).toBeVisible();
    await expect(figure.locator('.mermaid-description')).toBeVisible();
    await figure.locator('summary').click();
    await expect(figure.locator('code')).toContainText('flowchart TB');
    await expect(figure.locator('code')).toBeVisible();
  }
  await context.close();
});

test('a malformed diagram retains its explanation without breaking other diagrams', async ({
  page,
}) => {
  await page.route(`**${article}`, async (route) => {
    const response = await route.fetch();
    const body = (await response.text()).replace('flowchart TB', 'not-a-diagram');
    await route.fulfill({ response, body });
  });
  await page.goto(article);
  await expect(page.locator('.mermaid-error').first()).toBeVisible();
  await expect(page.locator('.mermaid-description').first()).toBeVisible();
  await expect(page.locator('.mermaid-canvas svg')).toHaveCount(2);
});

test('renderer loads only on articles with diagrams and fails gracefully if unavailable', async ({
  page,
}) => {
  const rendererRequests: string[] = [];
  page.on('request', (request) => {
    if (/\/mermaid[^/]*\.js/.test(request.url())) rendererRequests.push(request.url());
  });
  await page.goto('/notes/gradient-descent/');
  await expect(page.locator('.katex-display').first()).toBeVisible();
  await expect(page.locator('pre.astro-code').first()).toBeVisible();
  expect(rendererRequests).toEqual([]);
  await page.route(/\/mermaid[^/]*\.js/, (route) => route.abort());
  await page.goto(article);
  await expect(page.locator('.mermaid-error:visible')).toHaveCount(3);
  await expect(page.locator('.mermaid-description').first()).toBeVisible();
});
