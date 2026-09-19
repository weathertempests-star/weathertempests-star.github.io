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
  name: '姓名待填',
  role: '個人簡介待填',
  description: '記錄學習的軌跡，整理研究的思考。這裡收錄部落格文章、學術研究筆記與個人履歷。',
  bio: '待填：用幾句話介紹你的背景、關注的問題，以及正在探索的方向。',
  interests: [],
  email: '',
  links: [],
  education: [],
  experience: [],
  publications: [],
  skills: [],
};
