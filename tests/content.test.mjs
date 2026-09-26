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

test('collection metadata, notes series ordering, and draft-only sites are validated', () => {
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
      '---\ntitle: No series\ndescription: Missing series metadata\ndate: 2026-09-25\n---\nContent',
    );
    const invalidSeriesResult = build();
    assert.notEqual(invalidSeriesResult.status, 0);
    assert.match(invalidSeriesResult.stdout + invalidSeriesResult.stderr, /series/);
    rmSync(invalidNote);
    const invalidPaper = join(scratch, 'src/content/blog/invalid-paper.md');
    writeFileSync(
      invalidPaper,
      '---\ntitle: Invalid optional paper\ndescription: Paper metadata must be complete when supplied\ndate: 2026-09-25\npaper:\n  title: Incomplete\n---\nContent',
    );
    const invalidPaperResult = build();
    assert.notEqual(invalidPaperResult.status, 0);
    assert.match(invalidPaperResult.stdout + invalidPaperResult.stderr, /paper/);
    rmSync(invalidPaper);
    for (const collection of ['blog', 'notes'])
      writeFileSync(
        join(scratch, 'src/content', collection, 'private.md'),
        '---\ntitle: SECRET_FIXTURE\ndescription: SECRET_FIXTURE\ndate: 2026-09-19\ndraft: true\nfeatured: true\n' +
          (collection === 'notes' ? 'series:\n  id: optee-from-zero\n  order: 1\n' : '') +
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

    rmSync(join(scratch, 'src/content'), { recursive: true });
    mkdirSync(join(scratch, 'src/content/notes'), { recursive: true });
    const note = (title, date, order, draft = false) =>
      `---\ntitle: ${title}\ndescription: ${title} description\ndate: ${date}\ndraft: ${draft}\nseries:\n  id: optee-from-zero\n  order: ${order}\n---\n${title}`;
    writeFileSync(
      join(scratch, 'src/content/notes/first.md'),
      note('First chapter', '2020-01-01', 1),
    );
    writeFileSync(
      join(scratch, 'src/content/notes/second.md'),
      note('Second chapter', '2030-01-01', 2),
    );
    writeFileSync(
      join(scratch, 'src/content/notes/draft.md'),
      note('Private chapter', '2040-01-01', 3, true),
    );
    const orderedResult = build();
    assert.equal(orderedResult.status, 0, orderedResult.stdout + orderedResult.stderr);
    const seriesHtml = readFileSync(
      join(scratch, 'dist/notes/series/optee-from-zero/index.html'),
      'utf8',
    );
    assert.ok(seriesHtml.indexOf('First chapter') < seriesHtml.indexOf('Second chapter'));
    assert.doesNotMatch(seriesHtml, /Private chapter/);
    assert.equal(existsSync(join(scratch, 'dist/notes/draft/index.html')), false);
    const firstChapter = readFileSync(join(scratch, 'dist/notes/first/index.html'), 'utf8');
    const secondChapter = readFileSync(join(scratch, 'dist/notes/second/index.html'), 'utf8');
    assert.match(firstChapter, /<nav class="chapter-pagination"/);
    assert.match(firstChapter, /<a rel="next" href="\/notes\/second\/">/);
    assert.match(secondChapter, /<nav class="chapter-pagination"/);
    assert.match(secondChapter, /<a rel="prev" href="\/notes\/first\/">/);
    writeFileSync(
      join(scratch, 'src/content/notes/duplicate.md'),
      note('Duplicate chapter', '2025-01-01', 2),
    );
    const duplicateResult = build();
    assert.notEqual(duplicateResult.status, 0);
    assert.match(duplicateResult.stdout + duplicateResult.stderr, /duplicate|order/i);
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
});
