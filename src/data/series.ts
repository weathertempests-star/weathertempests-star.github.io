export const seriesCatalog = {
  'optee-from-zero': {
    title: 'OP-TEE 從零開始',
    description:
      '從 TrustZone 與信任邊界理解 OP-TEE，再逐步走向 QEMU 開發、安全介面與漏洞研究。從第一篇建立共同的概念，沿篇章接續閱讀。',
    status: '持續整理',
  },
} as const;

export type SeriesId = keyof typeof seriesCatalog;
export const seriesIds = Object.keys(seriesCatalog) as [SeriesId, ...SeriesId[]];
