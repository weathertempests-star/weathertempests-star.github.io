import { getCollection, type CollectionEntry } from 'astro:content';
import { seriesCatalog, type SeriesId } from '../data/series';

export type Entry = CollectionEntry<'blog'> | CollectionEntry<'notes'>;
export type Collection = Entry['collection'];

export async function published<C extends Collection>(
  collection: C,
): Promise<CollectionEntry<C>[]> {
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

export function seriesUrl(id: SeriesId): string {
  return `/notes/series/${id}/`;
}

export async function publishedSeries() {
  const notes = await published('notes');
  return Object.entries(seriesCatalog).flatMap(([key, info]) => {
    const id = key as SeriesId;
    const chapters = notes
      .filter((entry) => entry.data.series.id === id)
      .sort((a, b) => a.data.series.order - b.data.series.order);
    const orders = new Set<number>();
    for (const chapter of chapters) {
      if (orders.has(chapter.data.series.order)) {
        throw new Error(`Duplicate series order: ${id} #${chapter.data.series.order}`);
      }
      orders.add(chapter.data.series.order);
    }
    return chapters.length ? [{ id, ...info, chapters }] : [];
  });
}

export type PublishedSeries = Awaited<ReturnType<typeof publishedSeries>>[number];
