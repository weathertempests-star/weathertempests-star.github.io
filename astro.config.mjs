import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import { existsSync } from 'node:fs';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { resolveSite } from './src/lib/site-url.mjs';
import remarkMermaid from './src/lib/remark-mermaid.mjs';
import { migratedArticleSlugs } from './src/data/migrations.ts';

if (existsSync('.env')) process.loadEnvFile('.env');

export default defineConfig({
  site: resolveSite(process.env),
  output: 'static',
  trailingSlash: 'always',
  integrations: [
    sitemap({
      filter: (page) =>
        ![
          '/search/',
          '/404/',
          '/404.html',
          ...migratedArticleSlugs.map((slug) => `/notes/${slug}/`),
        ].includes(new URL(page).pathname),
    }),
  ],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkMermaid],
      rehypePlugins: [[rehypeKatex, { strict: 'ignore' }]],
    }),
    shikiConfig: { theme: 'github-light' },
  },
});
