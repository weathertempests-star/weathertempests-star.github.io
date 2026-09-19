/** Only root-level personal Pages sites are supported by this starter. */
export function resolveSite(env) {
  const repo = env.GITHUB_REPOSITORY;
  if (repo && env.GITHUB_ACTIONS === 'true') {
    const [owner, name] = repo.split('/');
    if (!owner || name?.toLowerCase() !== `${owner.toLowerCase()}.github.io`) {
      throw new Error('請將個人網站儲存庫命名為 <GitHub 帳號>.github.io，再執行部署。');
    }
    return `https://${owner.toLowerCase()}.github.io`;
  }
  const url = new URL(env.SITE_URL || 'http://localhost:4321');
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.pathname !== '/' ||
    url.search ||
    url.hash ||
    url.username ||
    url.password
  ) {
    throw new Error('SITE_URL 必須是 http(s) 根網址，例如 https://username.github.io');
  }
  return url.origin;
}
