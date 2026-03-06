# WeChat Article Exporter

将微信公众号文章导出为本地 HTML / Markdown，保留图片和数学公式。

## 功能特点

- 自动下载微信文章原始 HTML 并提取正文内容
- 下载文中所有图片到本地，替换远程链接为本地路径
- 生成两种 HTML 格式：本地图片引用版 + base64 内嵌单文件版
- HTML 转 Markdown，支持 LaTeX 公式、图片、列表、链接等
- 修复微信懒加载图片标签，确保离线可查看

## 项目结构

```
.
├── convert_wechat.sh          # 主 pipeline（推荐使用）
├── convert.py                 # HTML → Markdown 转换器
├── export_wechat_article.sh   # 备选 pipeline（Codex 方案，含 pandoc/weasyprint）
├── docs/
│   └── HTML_EXPORT_METHOD.md  # Codex 方案的方法文档
└── output/                    # 导出结果目录
```

## 依赖

**必需：**

- `bash`
- `curl`
- `rg`（[ripgrep](https://github.com/BurntSushi/ripgrep)）— 仅 `convert_wechat.sh` 使用
- `perl`
- `python3`（Python 3 标准库，无需额外安装）

**可选（仅 `export_wechat_article.sh` 使用）：**

- `pandoc` — HTML 转 Markdown
- `weasyprint` — HTML 转 PDF

## 快速开始

```bash
./convert_wechat.sh "https://mp.weixin.qq.com/s/xxxxx"
```

可指定输出目录名：

```bash
./convert_wechat.sh "https://mp.weixin.qq.com/s/xxxxx" my_article
```

输出目录结构：

```
output/my_article/
├── raw.html                # 原始下载的 HTML
├── content.html            # 提取的正文 HTML 片段
├── img_urls.txt            # 去重后的图片 URL 列表
├── image_map.tsv           # URL → 本地路径映射
├── images/                 # 下载的图片文件
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
├── article.static.html     # 本地图片路径的完整 HTML
├── article.embedded.html   # 图片 base64 内嵌的单文件 HTML（推荐查看）
└── article.md              # Markdown（LaTeX 公式 + 本地图片引用）
```

## 两套 Pipeline 对比

| | `convert_wechat.sh` | `export_wechat_article.sh` |
|---|---|---|
| Markdown 转换 | 自定义 `convert.py`（保留 LaTeX 公式） | `pandoc`（GFM 格式） |
| PDF 支持 | 无 | `weasyprint` |
| 额外依赖 | `rg` (ripgrep) | `pandoc`, `weasyprint` |
| 输出目录命名 | 自定义或时间戳 | 从 URL 提取文章 ID |
| 适用场景 | 含数学公式的技术文章 | 通用文章归档 |

两者的 HTML 导出流程（下载、提取正文、下载图片、替换路径、生成内嵌版）基本一致。

## convert.py 转换细节

`convert.py` 是自定义的 HTML → Markdown 转换器，针对微信公众号文章做了以下适配：

- **数学公式**：识别 `data-formula` 属性，短公式转 `$...$`，长公式转 `$$...$$` 块
- **列表内公式**：自动将列表中的 `$$` 块转为 `$\displaystyle ...$` 行内形式（兼容 VSCode Markdown 预览）
- **图片映射**：通过 `image_map.tsv` 将远程 URL 替换为本地 `images/` 路径
- **参考文献**：自动格式化脚注 `[^1]` 和参考资料分隔线
- **SVG 跳过**：跳过 SVG 标签内容（公式渲染的 SVG 由 LaTeX 源码替代）

## 查看导出结果

- **推荐**：浏览器直接打开 `article.embedded.html`（单文件，无外部依赖）
- **备选**：打开 `article.static.html`（需与 `images/` 目录保持相对路径）
- **Markdown**：使用支持 LaTeX 的编辑器打开 `article.md`（如 VSCode + Markdown Preview Enhanced）

## 常见问题

**图片显示为占位符？**
确认打开的是 `article.static.html` 或 `article.embedded.html`，而非中间文件。如仍有问题，检查 `image_map.tsv` 中映射数量是否与 `img_urls.txt` 行数一致。

**触发微信风控验证页？**
脚本会检测 `js_content` 是否存在。若报错 "未找到 js_content"，说明微信返回了验证页面，可稍后重试或更换网络。

**Markdown 公式显示异常？**
微信文章的公式存储为 SVG 或 `data-formula` 属性。`convert.py` 仅处理 `data-formula`，部分纯 SVG 公式可能丢失。建议以 HTML 版本为准。
