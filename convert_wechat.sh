#!/bin/bash
# ============================================================
# 微信公众号文章 → 本地 HTML + Markdown 转换脚本
# ============================================================
#
# 输出:
#   article.static.html   - 本地图片路径的完整 HTML（公式 SVG 原样保留）
#   article.embedded.html - 图片 base64 内嵌的单文件 HTML
#   article.md            - Markdown（LaTeX 公式 + 本地图片引用）
#
# 流程：
#   Step 1: curl 下载原始 HTML（移动端微信 UA）
#   Step 2: perl 提取 js_content 正文块
#   Step 3: 提取图片 URL + curl 下载到本地
#   Step 4: 生成 article.static.html（本地图片路径）
#   Step 5: 生成 article.embedded.html（base64 内嵌）
#   Step 6: convert.py 将 HTML 转为 Markdown
#
# 用法：
#   ./convert_wechat.sh <微信文章URL> [输出目录名]
#
# 依赖：curl, rg (ripgrep), perl, python3
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
URL="${1:?用法: $0 <微信文章URL> [输出目录名]}"
DIRNAME="${2:-wechat_$(date +%Y%m%d_%H%M%S)}"
OUTDIR="$SCRIPT_DIR/output/$DIRNAME"

UA='Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.53'

mkdir -p "$OUTDIR/images"

echo "=== Step 1: 下载 HTML ==="
curl -sL --compressed --max-time 30 \
  -A "$UA" \
  -H 'Referer: https://mp.weixin.qq.com/' \
  "$URL" -o "$OUTDIR/raw.html"
echo "  Downloaded: $(wc -c < "$OUTDIR/raw.html") bytes"

if ! rg -q 'id="js_content"' "$OUTDIR/raw.html"; then
  echo "ERROR: 未找到 js_content，可能触发了风控验证页。" >&2
  exit 1
fi

# 提取标题
TITLE=$(perl -0777 -ne '
  if (/<meta[^>]*property=(["\x27])og:title\1[^>]*content=(["\x27])([^"\x27]+)\2/is) {
    print $3; exit;
  }
  if (/var\s+msg_title\s*=\s*["\x27]([^"\x27]+)/s) {
    print $1; exit;
  }
' "$OUTDIR/raw.html")
TITLE="${TITLE:-WeChat Article}"
echo "  Title: $TITLE"

echo "=== Step 2: 提取正文块 ==="
# 使用与 codex 相同的精确 div 嵌套解析
perl -0777 -e '
  use strict; use warnings;
  my $f = shift @ARGV;
  open my $fh, "<:raw", $f or die "open: $!";
  local $/; my $html = <$fh>; close $fh;
  if ($html !~ m{<div\b[^>]*\bid=(["\x27])js_content\1[^>]*>}is) {
    die "Cannot find div#js_content\n";
  }
  my $open_end = $+[0];
  pos($html) = $open_end;
  my $depth = 1; my $inner_end;
  while ($html =~ m{<div\b[^>]*>|</div>}ig) {
    if ($& =~ m{^</div>}i) { $depth--; if ($depth == 0) { $inner_end = $-[0]; last; } }
    else { $depth++; }
  }
  die "Failed to parse div nesting\n" if !defined $inner_end;
  print substr($html, $open_end, $inner_end - $open_end);
' "$OUTDIR/raw.html" > "$OUTDIR/content.html"
echo "  Extracted: $(wc -c < "$OUTDIR/content.html") bytes"

echo "=== Step 3: 提取并下载图片 ==="
# 提取所有 mmbiz.qpic.cn 资源 URL（包括 data-src 和 src）
perl -0777 -ne '
  while (m{https?://mmbiz\.qpic\.cn/[^\s"\x27<>)]+}ig) {
    my $u = $&; $u =~ s/&amp;/&/g; $u =~ s/["\x27]$//g;
    print "$u\n" if $u =~ m{^https?://}i;
  }
' "$OUTDIR/content.html" | awk 'NF && !seen[$0]++' > "$OUTDIR/img_urls.txt"
IMG_COUNT=$(wc -l < "$OUTDIR/img_urls.txt" | tr -d ' ')
echo "  Found $IMG_COUNT unique images"

# 下载图片并记录映射
: > "$OUTDIR/image_map.tsv"
n=1
while IFS= read -r url; do
  [[ -z "$url" ]] && continue
  # 检测图片格式
  ext=$(printf '%s' "$url" | perl -ne '
    if (/[\?&]wx_fmt=([a-zA-Z0-9]+)/) { print lc($1); exit; }
    if (m{\.([a-zA-Z0-9]{2,5})(?:\?|$)}) { print lc($1); exit; }
    print "png";
  ')
  [[ "$ext" == "jpeg" ]] && ext="jpg"
  fname=$(printf "img_%03d.%s" "$n" "$ext")
  curl -sL --fail --retry 2 \
    -A "$UA" \
    -H 'Referer: https://mp.weixin.qq.com/' \
    "$url" -o "$OUTDIR/images/$fname" && {
    printf '%s\t%s\n' "$url" "images/$fname" >> "$OUTDIR/image_map.tsv"
    echo "  $fname: $(wc -c < "$OUTDIR/images/$fname" | tr -d ' ') bytes"
    n=$((n+1))
  } || echo "  WARN: failed to download: $url"
done < "$OUTDIR/img_urls.txt"

echo "=== Step 4: 生成 article.static.html ==="
# 先包装成完整 HTML 文档，再替换远程图片 URL 为本地路径
perl -e '
  use strict; use warnings;
  my ($map_file, $content_file, $out_file, $title) = @ARGV;

  # 读取 URL -> 本地路径映射
  my %map;
  open my $mf, "<", $map_file or die "open map: $!";
  while (<$mf>) {
    chomp; next if $_ eq "";
    my ($url, $local) = split /\t/, $_, 2;
    next if !defined $local;
    $map{$url} = $local;
    (my $esc = $url) =~ s/&/&amp;/g;
    $map{$esc} = $local;
  }
  close $mf;

  # 读取正文 HTML
  open my $fh, "<:raw", $content_file or die "open content: $!";
  local $/; my $html = <$fh>; close $fh;

  # 替换远程 URL 为本地路径
  for my $u (sort { length($b) <=> length($a) } keys %map) {
    my $l = $map{$u};
    $html =~ s/\Q$u\E/$l/g;
  }

  # 修复懒加载: 确保有 data-src 的 img 也有有效的 src
  $html =~ s{
    <img\b([^>]*?)\bdata-src=(["\x27])([^"\x27]+)\2([^>]*?)>
  }{
    my ($pre, $q, $u, $post) = ($1, $2, $3, $4);
    my $tag = "<img$pre data-src=$q$u$q$post>";
    if ($tag =~ /(?:^|\s)src\s*=\s*(["\x27])\s*\1/i) {
      $tag =~ s/(?:^|\s)src\s*=\s*(["\x27])\s*\1/ src="$u"/i;
    } elsif ($tag !~ /(?:^|\s)src\s*=/i) {
      $tag = "<img$pre src=\"$u\" data-src=$q$u$q$post>";
    }
    $tag;
  }geisx;

  # 输出完整 HTML 文档
  open my $of, ">:raw", $out_file or die "write: $!";
  print {$of} qq{<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$title</title>
  <style>
    body { max-width: 920px; margin: 24px auto; padding: 0 16px; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.75; }
    img { max-width: 100%; height: auto; display: block; margin: 12px auto; }
    pre { overflow-x: auto; }
  </style>
</head>
<body>
<h1>$title</h1>
};
  print {$of} $html;
  print {$of} "\n</body>\n</html>\n";
  close $of;
' "$OUTDIR/image_map.tsv" "$OUTDIR/content.html" "$OUTDIR/article.static.html" "$TITLE"

STATIC_SIZE=$(wc -c < "$OUTDIR/article.static.html" | tr -d ' ')
echo "  Generated: article.static.html ($STATIC_SIZE bytes)"

echo "=== Step 5: 生成 article.embedded.html ==="
perl -MMIME::Base64 -MFile::Basename -e '
  use strict; use warnings;
  my ($in_file, $out_file) = @ARGV;
  my $base = dirname($in_file);

  open my $fh, "<:raw", $in_file or die "open: $!";
  local $/; my $html = <$fh>; close $fh;

  sub mime {
    my ($p) = @_;
    return "image/png"  if $p =~ /\.png$/i;
    return "image/jpeg" if $p =~ /\.(jpg|jpeg)$/i;
    return "image/gif"  if $p =~ /\.gif$/i;
    return "image/webp" if $p =~ /\.webp$/i;
    return "application/octet-stream";
  }

  sub to_data_uri {
    my ($base, $src) = @_;
    return undef if !defined $src || $src !~ m{^images/};
    my $path = "$base/$src";
    return undef if !-f $path;
    local $/;
    open my $fh, "<:raw", $path or return undef;
    my $bin = <$fh>; close $fh;
    return "data:" . mime($path) . ";base64," . encode_base64($bin, "");
  }

  # 替换 img src 为 data URI
  $html =~ s{<img\b[^>]*>}{
    my $tag = $&;
    if ($tag =~ /(?:^|\s)src\s*=\s*"([^"]+)"/i) {
      my $src = $1;
      my $data = to_data_uri($base, $src);
      if (defined $data) {
        $tag =~ s/(?:^|\s)src\s*=\s*"[^"]+"/ src="$data"/i;
        $tag =~ s/\sdata-src\s*=\s*"[^"]*"//ig;
      }
    }
    $tag;
  }geisx;

  open my $of, ">:raw", $out_file or die "write: $!";
  print {$of} $html; close $of;
' "$OUTDIR/article.static.html" "$OUTDIR/article.embedded.html"

EMBED_SIZE=$(wc -c < "$OUTDIR/article.embedded.html" | tr -d ' ')
echo "  Generated: article.embedded.html ($EMBED_SIZE bytes)"

echo "=== Step 6: HTML → Markdown 转换 ==="
python3 "$SCRIPT_DIR/convert.py" "$OUTDIR"

echo ""
echo "=== Done ==="
echo "  Output:    $OUTDIR/"
echo "  HTML:      $OUTDIR/article.static.html (本地图片)"
echo "  HTML:      $OUTDIR/article.embedded.html (内嵌图片，推荐)"
echo "  Markdown:  $OUTDIR/article.md"
echo "  Images:    $OUTDIR/images/"
