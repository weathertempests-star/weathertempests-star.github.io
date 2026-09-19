import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { profile } from '../data/profile';
import { published, entryUrl } from '../lib/content';

export async function GET(context: APIContext) {
  const entries = [...(await published('blog')), ...(await published('notes'))].sort(
    (a, b) => b.data.date.getTime() - a.data.date.getTime(),
  );
  return rss({
    title: profile.siteName,
    description: profile.description,
    site: context.site!,
    items: entries.map((entry) => ({
      title: entry.data.title,
      description: entry.data.description,
      pubDate: entry.data.date,
      link: entryUrl(entry),
      categories: entry.data.tags,
    })),
    customData: '<language>zh-TW</language>',
  });
}
