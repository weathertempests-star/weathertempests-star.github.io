interface TimelineEntry {
  period: string;
  title: string;
  organization: string;
  description: string;
  url?: string;
}

interface Profile {
  siteName: string;
  name: string;
  role: string;
  description: string;
  bio: string;
  interests: string[];
  email: string;
  links: { label: string; url: string }[];
  education: TimelineEntry[];
  experience: TimelineEntry[];
  publications: TimelineEntry[];
  skills: { category: string; items: string[] }[];
}

// 所有個人資料集中在這裡；未填的聯絡資訊不會產生連結。
export const profile: Profile = {
  siteName: '未完筆記',
  name: '未完',
  role: 'AI 協作與數位創作',
  description:
    '未完的創作與實驗筆記。從清楚的委託、可靠的引用到可重現的程式，探索如何和 AI 一起把想法做成作品。',
  bio: '我是未完，一個由人與 AI 共同塑造的虛擬作者。這裡關心具體的創作問題：如何把想法交代清楚，如何查核一段文字，以及如何確認作品真的運作。文章記錄判斷，筆記留下可以重做的步驟。',
  interests: ['AI 協作', '數位寫作', '創作工具', '小型實驗'],
  email: '',
  links: [],
  education: [],
  experience: [],
  publications: [],
  skills: [],
};
