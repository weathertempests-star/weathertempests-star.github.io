import { readFileSync } from 'node:fs';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkMermaid from '../../src/lib/remark-mermaid.mjs';

// Reuse the built layout and its real assets. Only Playwright serves these pages;
// they never enter the content loader, search index, sitemap or Pages artifact.
export async function readingFixture(diagrams = true) {
  const processor = await createMarkdownProcessor({
    remarkPlugins: [remarkMath, remarkMermaid],
    rehypePlugins: [[rehypeKatex, { strict: 'ignore' }]],
  });
  const markdown = diagrams
    ? [1, 2, 3]
        .map((number) =>
          [
            '```mermaid',
            'flowchart TB',
            `accTitle: 測試流程 ${number}`,
            'accDescr: 請求送至服務後取得結果。',
            'request[請求] --> service[服務] --> result[結果]',
            '```',
          ].join('\n'),
        )
        .join('\n\n')
    : '$$\nx^2 + y^2 = z^2\n$$\n\n```js\nconst answer = 42;\n```';
  const { code } = await processor.render(markdown);
  const shell = readFileSync('dist/blog/learning-in-public/index.html', 'utf8');
  const article = /<article\b[^>]*data-pagefind-body[^>]*>[\s\S]*?<\/article>/;
  if (!article.test(shell)) throw new Error('Built article layout is missing');
  return shell.replace(article, () => `<article><div class="prose">${code}</div></article>`);
}
