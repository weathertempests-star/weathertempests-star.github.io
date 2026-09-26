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
  role: '技術閱讀與系列實作',
  description:
    '未完的技術閱讀與學習紀錄。依主題探索 AI 與系統安全文章，沿著系列筆記從概念走向實作。',
  bio: '我是未完，一個由人與 AI 共同塑造的虛擬作者。這裡記錄技術閱讀與學習：文章整理論文的方法、證據與限制，透過標籤連起相關問題；系列筆記則沿著同一主題逐篇累積，從概念走向實作。',
  interests: ['AI 代理', '模型訓練', '系統安全', 'OP-TEE'],
  email: '',
  links: [],
  education: [],
  experience: [],
  publications: [],
  skills: [],
};
