import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import remarkMermaid from '../src/lib/remark-mermaid.mjs';

test('Mermaid content stays escaped through the Markdown pipeline', async () => {
  const processor = await createMarkdownProcessor({ remarkPlugins: [remarkMermaid] });
  const { code } = await processor.render(
    [
      '```mermaid',
      'flowchart TB',
      'accTitle: <img src=x onerror=alert(1)>',
      'accDescr: A < B & B > C',
      'a["</code></pre><script>alert(1)</script>"]',
      '```',
    ].join('\n'),
  );
  assert.match(code, /class="mermaid-figure"/);
  assert.doesNotMatch(code, /<script>|<img\s/);
  assert.match(code, /&#x3C;script>|&lt;script&gt;/);
  assert.match(code, /A (?:&#x3C;|&lt;) B &#x26; B > C|A &lt; B &amp; B &gt; C/);
});

test('Mermaid diagrams missing a textual alternative fail the Markdown build', async () => {
  const processor = await createMarkdownProcessor({ remarkPlugins: [remarkMermaid] });
  for (const metadata of [
    '',
    'accTitle: Title',
    'accDescr: Description',
    'accTitle:   \naccDescr: Description',
    'accTitle: Title\naccDescr:   ',
  ]) {
    await assert.rejects(
      processor.render(`\`\`\`mermaid\nflowchart TB\n${metadata}\na --> b\n\`\`\``),
      /accTitle.*accDescr/,
    );
  }
});
