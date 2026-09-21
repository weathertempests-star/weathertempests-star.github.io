const escapeHtml = (value) =>
  value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

// Turn diagram fences into progressive-enhancement markup before syntax highlighting.
export default function remarkMermaid() {
  return (tree, file) => {
    const visit = (node) => {
      if (node.type === 'code' && node.lang === 'mermaid') {
        const source = node.value;
        const title = source.match(/^[ \t]*accTitle:[ \t]*(.*)$/m)?.[1].trim();
        const description = source.match(/^[ \t]*accDescr:[ \t]*(.*)$/m)?.[1].trim();
        if (!title || !description) {
          file.fail('Mermaid 圖表需要單行 accTitle 與 accDescr，供圖說與文字替代使用。', node);
        }
        node.type = 'html';
        node.value = `<figure class="mermaid-figure">
<figcaption>${escapeHtml(title)}</figcaption>
<div class="mermaid-canvas" hidden data-pagefind-ignore></div>
<p class="mermaid-description">${escapeHtml(description)}</p>
<p class="mermaid-error" role="status" hidden>圖表暫時無法顯示，仍可閱讀上方說明或查看原始碼。</p>
<details class="mermaid-source" data-pagefind-ignore><summary>檢視圖表原始碼</summary><pre><code>${escapeHtml(source)}</code></pre></details>
</figure>`;
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}
