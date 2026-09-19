import { readFileSync, readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

function hasArticles(directory) {
  return readdirSync(directory, { withFileTypes: true }).some((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory()
      ? entry.name !== 'pagefind' && hasArticles(path)
      : entry.name.endsWith('.html') &&
          /<article\b[^>]*\bdata-pagefind-body[\s=>]/.test(readFileSync(path, 'utf8'));
  });
}

if (!hasArticles('dist')) {
  rmSync('dist/pagefind', { recursive: true, force: true });
  console.log('尚無已發布文章，略過搜尋索引。');
} else {
  const runner = fileURLToPath(
    new URL('../node_modules/pagefind/lib/runner/bin.cjs', import.meta.url),
  );
  const result = spawnSync(
    process.execPath,
    [runner, '--site', 'dist', '--root-selector', '[data-pagefind-body]'],
    { stdio: 'inherit' },
  );
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}
