import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { resolveSite } from '../src/lib/site-url.mjs';

test('deployment uses the real personal Pages origin and rejects project repositories', () => {
  assert.equal(resolveSite({}), 'http://localhost:4321');
  assert.equal(
    resolveSite({ SITE_URL: 'https://example.github.io/' }),
    'https://example.github.io',
  );
  assert.equal(
    resolveSite({ GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: 'Alice/alice.github.io' }),
    'https://alice.github.io',
  );
  assert.throws(
    () => resolveSite({ GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: 'Alice/blog' }),
    /github.io/,
  );
  for (const SITE_URL of [
    'https://example.com/blog/',
    'ftp://example.com',
    'https://example.com/?x=1',
  ]) {
    assert.throws(() => resolveSite({ SITE_URL }));
  }
});

const root = resolve('dist');
function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}

test('static output contains required pages and search assets', () => {
  for (const path of [
    'index.html',
    'blog/index.html',
    'notes/index.html',
    'cv/index.html',
    'search/index.html',
    '404.html',
    'rss.xml',
    'sitemap-index.xml',
  ]) {
    assert.ok(existsSync(join(root, path)), `${path} is missing; run npm run build first`);
  }
  if (
    files(root)
      .filter((path) => path.endsWith('.html'))
      .some((path) => readFileSync(path, 'utf8').includes('<article data-pagefind-body'))
  ) {
    assert.ok(existsSync(join(root, 'pagefind/pagefind.js')));
  }
});

test('all internal links, assets and heading anchors resolve in the static build', () => {
  const htmlFiles = files(root).filter((path) => path.endsWith('.html'));
  for (const path of htmlFiles) {
    const html = readFileSync(path, 'utf8');
    const canonical = html.match(/rel="canonical" href="([^"]+)"/)?.[1];
    assert.ok(canonical, `Missing canonical: ${path}`);
    for (const [, raw] of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
      const value = raw.replaceAll('&amp;', '&');
      if (!value.startsWith('/') && !value.startsWith('#')) continue;
      const url = new URL(value, canonical);
      let target = value.startsWith('#') ? path : join(root, decodeURIComponent(url.pathname));
      if (!value.startsWith('#') && url.pathname.endsWith('/')) target = join(target, 'index.html');
      assert.ok(existsSync(target), `Broken link ${value} in ${path}`);
      if (url.hash && target.endsWith('.html')) {
        const targetHtml = readFileSync(target, 'utf8');
        assert.ok(
          targetHtml.includes(`id="${decodeURIComponent(url.hash.slice(1))}"`),
          `Broken heading ${value} in ${path}`,
        );
      }
    }
  }
});
