import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Without Diresction",
    pageTitleSuffix: "無向",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "zh-TW",
    baseUrl: "weathertempests-star.github.io",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Noto Serif TC",   // 中文標題字體（視覺較優雅）
        body: "Noto Sans TC",      // 中文內文
        code: "JetBrains Mono",
      },
      colors: {
      lightMode: {
        light: "#faf8f6",        // 暖米白背景（比純白柔和）
        lightgray: "#e5e0d8",
        gray: "#b8b0a8",
        darkgray: "#3d3530",
        dark: "#1a1410",
        secondary: "#7c6f64",    // Gruvbox 風格中性棕
        tertiary: "#a89984",
        highlight: "rgba(143, 134, 120, 0.15)",
        textHighlight: "#f2e5bc88",
      },
      darkMode: {
        light: "#1d2021",        // Gruvbox Dark 背景
        lightgray: "#3c3836",
        gray: "#665c54",
        darkgray: "#d5c4a1",
        dark: "#fbf1c7",
        secondary: "#83a598",
        tertiary: "#689d6a",
        highlight: "rgba(131, 165, 152, 0.15)",
        textHighlight: "#b5761388",
      },
    },
    }
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ 
        enableInHtmlEmbed: false,
        parseTags: true,
        parseArrows: false,
        parseBlockReferences: true,
      }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents({
        maxDepth: 3,           // 目錄最深到 H3
        minEntries: 3,         // 至少 3 個標題才顯示目錄
        showByDefault: true,
        collapseByDefault: false,
      }),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
        rssLimit: 20,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
