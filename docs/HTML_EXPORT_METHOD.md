# 微信文章导出为本地 HTML（含图片）完整方法

本文只记录 HTML 导出流程，不讨论 Markdown 质量问题。

## 1. 目标
把这篇微信文章完整导出为本地可查看 HTML，并保留文中图片：

- 原文链接: `https://mp.weixin.qq.com/s/YA5akqX4oZfc2MScjzRDDQ`
- 推荐查看文件: `article.embedded.html`（单文件，图片已内嵌）
- 备选查看文件: `article.static.html`（依赖同目录 `images/`）

## 2. 前置依赖
需要以下命令可用：

- `bash`
- `curl`
- `perl`
- `awk`
- `find`
- `wc`

可选：

- `pandoc`（仅用于顺便生成 `article.md`，与 HTML 导出无关）
- `weasyprint`（用于 HTML 转 PDF）

## 3. 一键执行
在仓库根目录运行：

```bash
./exports/codex/export_wechat_article.sh \
  "https://mp.weixin.qq.com/s/YA5akqX4oZfc2MScjzRDDQ"
```

说明：

- 默认输出到 `exports/codex/output/<文章ID>/`，例如本篇为
  `exports/codex/output/wechat_YA5akqX4oZfc2MScjzRDDQ/`。
- 也可以手动指定输出目录；指定后脚本会先清理旧产物再重跑。
- 抓取使用移动端微信 UA，并带 `Referer: https://mp.weixin.qq.com/`。

## 4. 脚本实际做了什么
按执行顺序：

1. 抓原始页面到 `raw_page.html`。
2. 从原始 HTML 里解析 `div#js_content` 到 `js_content.inner.html`。
3. 包装成可独立打开的 `js_content.html`。
4. 提取 `mmbiz.qpic.cn` 资源链接，去重后写入 `image_urls.txt`。
5. 下载资源到 `images/`，并记录 `image_map.tsv` 映射。
6. 把 HTML 里的远程图片 URL 替换为本地相对路径，生成 `article.static.html`。
7. 修复懒加载标签：确保图片标签有可显示的 `src`。
8. 生成 `article.embedded.html`：把 `images/...` 转成 base64 data URI。
9. 记录运行日志 `run.log` 和流程文件 `PROCESS.md`。

## 5. 如何验证“HTML 是否完整”
在仓库根目录运行：

```bash
out=exports/codex/output/wechat_YA5akqX4oZfc2MScjzRDDQ
wc -l "$out/image_urls.txt" "$out/image_map.tsv"
find "$out/images" -type f | wc -l
```

期望三个数字一致（当前为 32/32/32）。

继续检查 HTML 图片引用：

```bash
out=exports/codex/output/wechat_YA5akqX4oZfc2MScjzRDDQ
perl -0777 -ne 'my $img=()=/<img\b/ig; my $src_local=()=/<img\b[^>]*\bsrc\s*=\s*["\x27]images\//ig; my $src_empty=()=/<img\b[^>]*\bsrc\s*=\s*["\x27]\s*["\x27]/ig; print "static img=$img src_local=$src_local src_empty=$src_empty\n";' "$out/article.static.html"
perl -0777 -ne 'my $img=()=/<img\b/ig; my $src_data=()=/<img\b[^>]*\bsrc\s*=\s*["\x27]data:image\//ig; my $src_empty=()=/<img\b[^>]*\bsrc\s*=\s*["\x27]\s*["\x27]/ig; print "embedded img=$img src_data=$src_data src_empty=$src_empty\n";' "$out/article.embedded.html"
```

当前结果：

- `article.static.html`: `img=26 src_local=23 src_empty=1`
- `article.embedded.html`: `img=26 src_data=23 src_empty=1`

其中 `src_empty=1` 是微信模板占位图，不属于正文缺图。

## 6. 打开方式
本地浏览器直接打开：

- `exports/codex/output/wechat_YA5akqX4oZfc2MScjzRDDQ/article.embedded.html`（优先）
- `exports/codex/output/wechat_YA5akqX4oZfc2MScjzRDDQ/article.static.html`（需要 `images/` 同目录）

## 7. HTML 转 PDF（weasyprint）
在仓库根目录运行：

```bash
out=exports/codex/output/wechat_YA5akqX4oZfc2MScjzRDDQ
weasyprint "$out/article.static.html" "$out/article.static.weasy.pdf"
weasyprint "$out/article.embedded.html" "$out/article.embedded.weasy.pdf"
```

说明：

- `article.static.weasy.pdf`：基于本地图片路径版 HTML 生成。
- `article.embedded.weasy.pdf`：基于内嵌图片版 HTML 生成。
- 一般两者视觉结果接近；优先保留 `embedded` 版本便于归档。

## 8. 常见失败点与处理
1. 图片仍显示占位符。
原因：文件不是脚本输出的最终 `article.static.html` 或 `article.embedded.html`。
处理：确认打开的路径正确，重新运行第 3 节命令。

2. 图片下载数量偏少。
原因：网络波动或微信侧限制。
处理：重跑一遍脚本，检查 `run.log` 是否有 `WARN: failed to download image`。

3. 只有 HTML 正常，Markdown 异常。
原因：Markdown 转换是降级路径，对微信富文本和公式兼容性不足。
处理：以 HTML 作为归档与阅读主文件。
