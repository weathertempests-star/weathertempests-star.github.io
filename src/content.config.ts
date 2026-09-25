import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const schema = z.object({
  title: z.string().trim().min(1),
  description: z.string().trim().min(1),
  date: z.coerce.date(),
  tags: z.array(z.string().trim().min(1)).default([]),
  draft: z.boolean().default(false),
  featured: z.boolean().default(false),
  sample: z.boolean().default(false),
});

export const collections = {
  blog: defineCollection({
    loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
    schema,
  }),
  notes: defineCollection({
    loader: glob({ pattern: '**/*.md', base: './src/content/notes' }),
    schema: schema.extend({
      paper: z.object({
        title: z.string().trim().min(1),
        authors: z.array(z.string().trim().min(1)).min(1),
        url: z.url({ protocol: /^https?$/ }),
        published: z.coerce.date(),
        status: z.enum(['preprint', 'published']),
        version: z.string().trim().min(1).optional(),
        venue: z.string().trim().min(1).optional(),
      }),
    }),
  }),
};
