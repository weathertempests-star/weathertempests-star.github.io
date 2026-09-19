import { getCollection, type CollectionEntry } from 'astro:content';

export type Entry = CollectionEntry<'blog'> | CollectionEntry<'notes'>;
export type Collection = Entry['collection'];

export async function published(collection: Collection): Promise<Entry[]> {
  return (await getCollection(collection, ({ data }) => !data.draft)).sort(
    (a, b) => b.data.date.getTime() - a.data.date.getTime() || a.id.localeCompare(b.id),
  );
}

export function entryUrl(entry: Entry): string {
  return `/${entry.collection}/${entry.id.split('/').map(encodeURIComponent).join('/')}/`;
}

export function formatDate(date: Date): string {
  return new Intl.DateTimeFormat('zh-TW', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: 'Asia/Taipei',
  }).format(date);
}

export function readingTime(body = ''): number {
  const chinese = body.match(/[\u3400-\u9FFF]/g)?.length ?? 0;
  const words = body.replace(/[\u3400-\u9FFF]/g, '').match(/[A-Za-z0-9]+/g)?.length ?? 0;
  return Math.max(1, Math.ceil(chinese / 400 + words / 200));
}
