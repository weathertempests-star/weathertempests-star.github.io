import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/jackyzha0/quartz",
      "Discord Community": "https://discord.gg/cRFFHYye7t",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),       // 顯示路徑：Blog > Deep Learning > EP0
    Component.ArticleTitle(),
    Component.ContentMeta(),       // 顯示日期、字數、閱讀時間
    Component.TagList(),           // 文章 tag
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.DesktopOnly(
      Component.Explorer({
        title: "Explorer",
        folderClickBehavior: "collapse",   // 點擊資料夾展開/收合
        folderDefaultState: "collapsed",   // 預設收合（畫面更乾淨）
        useSavedState: true,               // 記住使用者展開狀態
        // 隱藏 tags 資料夾（避免 Explorer 太雜亂）
        filterFn: (node) => node.name !== "tags",
      })
    ),
  ],
  right: [
    Component.Graph({              // 知識圖譜
      localGraph: {
        depth: 2,                  // 顯示二階鄰居
        showTags: false,
      },
    }),
    Component.DesktopOnly(Component.TableOfContents()),  // 文章目錄
    Component.Backlinks(),         // 反向連結（哪些文章引用了這篇）
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer(),
  ],
  right: [],
}
