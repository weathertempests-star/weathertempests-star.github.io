import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

test('missing required metadata fails clearly; a site containing only drafts remains empty', () => {
  const project = process.cwd();
  const scratch = mkdtempSync(join(tmpdir(), 'journal-content-'));
  try {
    for (const name of [
      'src',
      'public',
      'scripts',
      'astro.config.mjs',
      'tsconfig.json',
      'package.json',
    ])
      cpSync(resolve(name), join(scratch, name), { recursive: true });
    symlinkSync(join(project, 'node_modules'), join(scratch, 'node_modules'), 'dir');
    rmSync(join(scratch, 'src/content'), { recursive: true });
    for (const collection of ['blog', 'notes'])
      mkdirSync(join(scratch, 'src/content', collection), { recursive: true });
    const invalid = join(scratch, 'src/content/blog/invalid.md');
    writeFileSync(
      invalid,
      '---\ntitle: Invalid metadata\ndate: 2026-09-19\n---\nMissing description.',
    );
    const env = {
      ...process.env,
      ASTRO_TELEMETRY_DISABLED: '1',
      GITHUB_ACTIONS: '',
      SITE_URL: 'https://fixture.github.io',
    };
    const build = () =>
      spawnSync(process.execPath, [join(project, 'node_modules/astro/bin/astro.mjs'), 'build'], {
        cwd: scratch,
        env,
        encoding: 'utf8',
        timeout: 60000,
      });
    const invalidResult = build();
    assert.notEqual(invalidResult.status, 0);
    assert.match(invalidResult.stdout + invalidResult.stderr, /description/);
    rmSync(invalid);
    const invalidNote = join(scratch, 'src/content/notes/invalid.md');
    writeFileSync(
      invalidNote,
      '---\ntitle: No paper\ndescription: Missing source\ndate: 2026-09-25\n---\nContent',
    );
    const invalidPaperResult = build();
    assert.notEqual(invalidPaperResult.status, 0);
    assert.match(invalidPaperResult.stdout + invalidPaperResult.stderr, /paper/);
    rmSync(invalidNote);
    for (const collection of ['blog', 'notes'])
      writeFileSync(
        join(scratch, 'src/content', collection, 'private.md'),
        '---\ntitle: SECRET_FIXTURE\ndescription: SECRET_FIXTURE\ndate: 2026-09-19\ndraft: true\nfeatured: true\n' +
          (collection === 'notes'
            ? 'paper:\n  title: SECRET_FIXTURE\n  authors: [SECRET_FIXTURE]\n  url: https://example.com/paper\n  published: 2026-09-01\n  status: preprint\n'
            : '') +
          '---\nSECRET_FIXTURE',
      );
    const emptyResult = build();
    assert.equal(emptyResult.status, 0, emptyResult.stdout + emptyResult.stderr);
    for (const collection of ['blog', 'notes'])
      assert.equal(existsSync(join(scratch, 'dist', collection, 'private/index.html')), false);
    for (const path of [
      'index.html',
      'blog/index.html',
      'notes/index.html',
      'cv/index.html',
      'rss.xml',
      'sitemap-0.xml',
    ])
      assert.doesNotMatch(
        readFileSync(join(scratch, 'dist', path), 'utf8'),
        /SECRET_FIXTURE|\/private\//,
      );
    const output = join(scratch, 'dist');
    for (const path of readdirSync(output, { recursive: true }).filter((path) =>
      path.endsWith('.html'),
    )) {
      const html = readFileSync(join(output, path), 'utf8');
      for (const [, href] of html.matchAll(/\bhref="([^"]+)"/g)) {
        const url = new URL(href.replaceAll('&amp;', '&'), `${env.SITE_URL}/${path}`);
        if (url.origin !== env.SITE_URL || !/^\/(?:blog|notes)\/[^/]+/.test(url.pathname)) continue;
        const target = join(output, decodeURIComponent(url.pathname));
        assert.ok(
          existsSync(url.pathname.endsWith('.html') ? target : join(target, 'index.html')),
          `Unpublished article link ${href} in ${path}`,
        );
      }
    }
    assert.match(readFileSync(join(scratch, 'dist/search/index.html'), 'utf8'), /還沒有已發布/);
    const indexed = spawnSync(process.execPath, [join(scratch, 'scripts/build-search.mjs')], {
      cwd: scratch,
      env,
      encoding: 'utf8',
      timeout: 60000,
    });
    assert.equal(indexed.status, 0, indexed.stdout + indexed.stderr);
    assert.match(indexed.stdout, /略過搜尋索引/);
    assert.equal(existsSync(join(scratch, 'dist/pagefind')), false);
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
});
