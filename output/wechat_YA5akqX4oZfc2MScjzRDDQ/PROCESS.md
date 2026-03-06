# WeChat 导出流程记录

- 运行时间: 2026-03-05 17:24:38 +0800
- 原始链接: https://mp.weixin.qq.com/s/YA5akqX4oZfc2MScjzRDDQ
- 输出目录: ./exports/codex/output/wechat_YA5akqX4oZfc2MScjzRDDQ

## 流程
1. 使用移动端微信 UA 抓取原始网页，保存为 `raw_page.html`。
2. 从原始网页中解析并提取 `div#js_content` 内容，生成 `js_content.html`。
3. 提取 `js_content` 中所有 `mmbiz.qpic.cn` 图片资源 URL，去重后下载到 `images/`。
4. 生成 URL 到本地文件映射 `image_map.tsv`。
5. 将文章 HTML 中图片地址替换为本地路径，并强制写入 `src`，生成 `article.static.html`。
6. 将本地图片进一步内嵌为 base64 data URI，生成 `article.embedded.html`。
7. （可选）使用 pandoc 将 `article.static.html` 转成 `article.md`。

## 产物
- `raw_page.html`
- `js_content.inner.html`
- `js_content.html`
- `image_urls.txt`
- `image_map.tsv`
- `images/` (下载图片总数: 32)
- `article.static.html` (本地图片 src 数: 23)
- `article.embedded.html` (内嵌图片 src 数: 23)
- `article.md`（若已安装 pandoc）
- `run.log`

## 打开方式
- 推荐离线查看: `article.embedded.html`（单文件，不依赖外部图片目录）
- 保持相对目录查看: `article.static.html`（需与 `images/` 同目录）
