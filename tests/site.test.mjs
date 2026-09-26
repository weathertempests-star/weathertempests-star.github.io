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
    'blog/evirca/index.html',
    'notes/index.html',
    'notes/series/optee-from-zero/index.html',
    'notes/optee-from-zero-01/index.html',
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
      .some((path) => /<article\b[^>]*\bdata-pagefind-body[\s=>]/.test(readFileSync(path, 'utf8')))
  ) {
    assert.ok(existsSync(join(root, 'pagefind/pagefind.js')));
  }
});

test('RSS and sitemap include published articles and preserve the public origin', () => {
  const htmlFiles = files(root).filter((path) => path.endsWith('.html'));
  const articleUrls = htmlFiles.flatMap((path) => {
    const html = readFileSync(path, 'utf8');
    if (!/<article\b[^>]*\bdata-pagefind-body[\s=>]/.test(html)) return [];
    const canonical = html.match(/rel="canonical" href="([^"]+)"/)?.[1];
    assert.ok(canonical, `Missing article canonical: ${path}`);
    return [canonical];
  });
  assert.ok(articleUrls.length > 0, 'The site must contain published articles');

  const rss = readFileSync(join(root, 'rss.xml'), 'utf8');
  const channel = rss.split('<item>')[0];
  assert.match(channel, /<title>\s*\S[\s\S]*?<\/title>/);
  assert.match(channel, /<description>\s*\S[\s\S]*?<\/description>/);
  const feedUrls = [...rss.matchAll(/<item>[\s\S]*?<link>([^<]+)<\/link>[\s\S]*?<\/item>/g)].map(
    ([, url]) => url.replaceAll('&amp;', '&'),
  );
  assert.deepEqual(feedUrls.toSorted(), articleUrls.toSorted());

  const sitemap = files(root)
    .filter((path) => /sitemap-\d+\.xml$/.test(path))
    .map((path) => readFileSync(path, 'utf8'))
    .join('');
  const sitemapUrls = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map(
    ([, url]) => new URL(url.replaceAll('&amp;', '&')),
  );
  const origin = new URL(articleUrls[0]).origin;
  for (const url of sitemapUrls) {
    assert.equal(url.origin, origin);
    assert.ok(!['/search/', '/404/', '/404.html'].includes(url.pathname));
  }
  for (const path of ['/', '/blog/', '/notes/', '/cv/']) {
    assert.ok(
      sitemapUrls.some((url) => url.pathname === path),
      `Missing sitemap entry: ${path}`,
    );
  }
  for (const article of articleUrls) {
    assert.ok(
      sitemapUrls.some((url) => url.href === article),
      `Missing article: ${article}`,
    );
    assert.equal(new URL(article).origin, origin);
  }
  assert.doesNotMatch(rss + sitemap, /PRIVATE_DRAFT_SENTINEL|\/draft-example\//);
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

test('blog contains ten paper articles; notes contain the OP-TEE series and safe legacy redirects', () => {
  const expected = [
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
  const blogPages = readdirSync(join(root, 'blog')).filter((name) =>
    statSync(join(root, 'blog', name)).isDirectory(),
  );
  assert.deepEqual(blogPages.sort(), expected.sort());
  for (const slug of expected) {
    const html = readFileSync(join(root, 'blog', slug, 'index.html'), 'utf8');
    assert.match(html, /class="paper-info"/);
    assert.match(html, /https:\/\/arxiv\.org\/abs\/\d+\.\d+v\d+/);
  }
  const notePages = readdirSync(join(root, 'notes')).filter((name) =>
    statSync(join(root, 'notes', name)).isDirectory(),
  );
  assert.deepEqual(notePages.sort(), [...expected, 'optee-from-zero-01', 'series'].sort());
  const series = readFileSync(join(root, 'notes/series/optee-from-zero/index.html'), 'utf8');
  assert.match(series, /OP-TEE 從零開始/);
  assert.match(series, /class="chapter-list"/);
  assert.match(series, /\/notes\/optee-from-zero-01\//);
  const chapter = readFileSync(join(root, 'notes/optee-from-zero-01/index.html'), 'utf8');
  assert.match(chapter, /class="series-navigation"/);
  assert.match(chapter, /aria-current="page"/);
  for (const slug of expected) {
    const html = readFileSync(join(root, 'notes', slug, 'index.html'), 'utf8');
    assert.match(html, /<meta(?=[^>]*name="robots")(?=[^>]*content="noindex\b)/);
    assert.match(html, new RegExp(`rel="canonical" href="[^"]*/blog/${slug}/"`));
    assert.match(html, new RegExp(`http-equiv="refresh" content="[^\"]*/blog/${slug}/`));
    assert.doesNotMatch(html, /data-pagefind-body/);
  }
  for (const path of files(root)) {
    assert.doesNotMatch(path, /(?:AGENT|daily paper digest|__test)(?:\/|$)/);
    if (!path.endsWith('.html') && !path.endsWith('.xml')) continue;
    assert.doesNotMatch(readFileSync(path, 'utf8'), /\/notes\/gradient-descent\//);
  }
  const indexes = [
    readFileSync(join(root, 'rss.xml'), 'utf8'),
    ...files(root)
      .filter((path) => /sitemap-\d+\.xml$/.test(path))
      .map((path) => readFileSync(path, 'utf8')),
  ].join('');
  for (const slug of expected) assert.doesNotMatch(indexes, new RegExp(`/notes/${slug}/`));
});
