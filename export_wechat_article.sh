#!/usr/bin/env bash
set -euo pipefail

# Export a WeChat article to local HTML/Markdown with images.
# Usage:
#   ./exports/codex/export_wechat_article.sh "<url>" [output_dir]

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <wechat_article_url> [output_dir]" >&2
  exit 1
fi

URL="$1"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
ARTICLE_KEY="$(
  perl -e '
    use strict;
    use warnings;
    my $u = shift @ARGV // "";
    if ($u =~ m{/s/([^/?#]+)}i) {
      my $k = $1;
      $k =~ s/[^A-Za-z0-9._-]+/_/g;
      print "wechat_$k";
      exit;
    }
    print "wechat_export_'$TS'";
  ' "$URL"
)"
OUT_DIR="${2:-$BASE_DIR/output/$ARTICLE_KEY}"
IMG_DIR="$OUT_DIR/images"

UA='Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.54'

mkdir -p "$OUT_DIR" "$IMG_DIR"

RAW_HTML="$OUT_DIR/raw_page.html"
JS_INNER="$OUT_DIR/js_content.inner.html"
JS_WRAPPED="$OUT_DIR/js_content.html"
URLS_RAW="$OUT_DIR/image_urls_raw.txt"
URLS_UNIQ="$OUT_DIR/image_urls.txt"
MAP_TSV="$OUT_DIR/image_map.tsv"
STATIC_HTML="$OUT_DIR/article.static.html"
EMBEDDED_HTML="$OUT_DIR/article.embedded.html"
MD_FILE="$OUT_DIR/article.md"
PROCESS_MD="$OUT_DIR/PROCESS.md"
RUN_LOG="$OUT_DIR/run.log"

# Clean previous outputs when reusing an existing directory.
rm -f "$RAW_HTML" "$JS_INNER" "$JS_WRAPPED" "$URLS_RAW" "$URLS_UNIQ" \
  "$MAP_TSV" "$STATIC_HTML" "$EMBEDDED_HTML" "$MD_FILE" "$PROCESS_MD" "$RUN_LOG"
find "$IMG_DIR" -type f -delete

exec > >(tee "$RUN_LOG") 2>&1

echo "[1/8] Fetching raw page..."
curl -L --fail --compressed "$URL" \
  -A "$UA" \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
  -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8' \
  -H 'Referer: https://mp.weixin.qq.com/' \
  -o "$RAW_HTML"

echo "[2/8] Extracting #js_content..."
perl -0777 -e '
  use strict;
  use warnings;
  my $f = shift @ARGV;
  open my $fh, "<:raw", $f or die "open failed: $!";
  local $/;
  my $html = <$fh>;
  close $fh;

  if ($html !~ m{<div\b[^>]*\bid=(["\x27])js_content\1[^>]*>}is) {
    die "Cannot find div#js_content\n";
  }
  my $open_start = $-[0];
  my $open_end = $+[0];
  pos($html) = $open_end;
  my $depth = 1;
  my $inner_end = undef;

  while ($html =~ m{<div\b[^>]*>|</div>}ig) {
    my $tok = $&;
    if ($tok =~ m{^</div>}i) {
      $depth--;
      if ($depth == 0) {
        $inner_end = $-[0];
        last;
      }
    } else {
      $depth++;
    }
  }

  die "Failed to parse div nesting for #js_content\n" if !defined $inner_end;
  my $inner = substr($html, $open_end, $inner_end - $open_end);
  print $inner;
' "$RAW_HTML" > "$JS_INNER"

TITLE="$(
  perl -0777 -ne '
    if (/<meta[^>]*property=(["\x27])og:title\1[^>]*content=(["\x27])([^"\x27]+)\2/is) {
      print $3; exit;
    }
    if (/<title[^>]*>(.*?)<\/title>/is) {
      my $t = $1; $t =~ s/^\s+|\s+$//g; print $t; exit;
    }
  ' "$RAW_HTML"
)"
if [[ -z "$TITLE" ]]; then
  TITLE="WeChat Article Export"
fi

cat > "$JS_WRAPPED" <<EOF
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${TITLE}</title>
  <style>
    body { max-width: 920px; margin: 24px auto; padding: 0 16px; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.75; }
    img { max-width: 100%; height: auto; display: block; margin: 12px auto; }
    pre { overflow-x: auto; }
  </style>
</head>
<body>
EOF
printf '<h1>%s</h1>\n' "$TITLE" >> "$JS_WRAPPED"
cat "$JS_INNER" >> "$JS_WRAPPED"
cat >> "$JS_WRAPPED" <<'EOF'
</body>
</html>
EOF

echo "[3/8] Extracting image URLs..."
perl -0777 -ne '
  while (m{https?://mmbiz\.qpic\.cn/[^\s"\x27<>)]+}ig) {
    my $u = $&;
    $u =~ s/&amp;/&/g;
    $u =~ s/&quot;$//ig;
    $u =~ s/[\"\x27]$//g;
    next unless $u =~ m{^https?://}i;
    print "$u\n";
  }
' "$JS_WRAPPED" > "$URLS_RAW"
awk 'NF && !seen[$0]++' "$URLS_RAW" > "$URLS_UNIQ"

echo "[4/8] Downloading images..."
: > "$MAP_TSV"
idx=1
while IFS= read -r url; do
  [[ -z "$url" ]] && continue
  ext="$(printf '%s' "$url" | perl -ne '
    if (/[\?&]wx_fmt=([a-zA-Z0-9]+)/) { print lc($1); exit; }
    if (m{\.([a-zA-Z0-9]{2,5})(?:\?|$)}) { print lc($1); exit; }
    print "png";
  ')"
  case "$ext" in
    jpeg) ext="jpg" ;;
    *) ;;
  esac
  local_rel="$(printf 'images/img_%03d.%s' "$idx" "$ext")"
  local_abs="$OUT_DIR/$local_rel"

  if curl -L --fail --retry 3 --retry-delay 1 \
    -A "$UA" \
    -H 'Referer: https://mp.weixin.qq.com/' \
    "$url" -o "$local_abs"; then
    printf '%s\t%s\n' "$url" "$local_rel" >> "$MAP_TSV"
    idx=$((idx + 1))
  else
    echo "WARN: failed to download image: $url"
  fi
done < "$URLS_UNIQ"

echo "[5/8] Rewriting image URLs to local files..."
perl -e '
  use strict;
  use warnings;

  my ($map_file, $in_file, $out_file) = @ARGV;
  my %map;
  open my $mf, "<", $map_file or die "open map failed: $!";
  while (<$mf>) {
    chomp;
    next if $_ eq "";
    my ($url, $local) = split /\t/, $_, 2;
    next if !defined $local;
    $map{$url} = $local;
    (my $esc = $url) =~ s/&/&amp;/g;
    $map{$esc} = $local;
  }
  close $mf;

  open my $fh, "<:raw", $in_file or die "open html failed: $!";
  local $/;
  my $html = <$fh>;
  close $fh;

  for my $u (sort { length($b) <=> length($a) } keys %map) {
    my $l = $map{$u};
    $html =~ s/\Q$u\E/$l/g;
  }

  # Ensure local images always have a non-empty src attribute.
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

  open my $of, ">:raw", $out_file or die "write html failed: $!";
  print {$of} $html;
  close $of;
' "$MAP_TSV" "$JS_WRAPPED" "$STATIC_HTML"

echo "[6/8] Building single-file embedded HTML..."
perl -MMIME::Base64 -MFile::Basename -e '
  use strict;
  use warnings;

  my ($in_file, $out_file) = @ARGV;
  my $base = dirname($in_file);
  open my $fh, "<:raw", $in_file or die "open html failed: $!";
  local $/;
  my $html = <$fh>;
  close $fh;

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
    return undef if !defined $src;
    return undef if $src !~ m{^images/};
    my $path = "$base/$src";
    return undef if !-f $path;
    local $/;
    open my $fh, "<:raw", $path or return undef;
    my $bin = <$fh>;
    close $fh;
    my $b64 = encode_base64($bin, "");
    my $m = mime($path);
    return "data:$m;base64,$b64";
  }

  $html =~ s{
    <img\b[^>]*>
  }{
    my $tag = $&;
    if ($tag =~ /(?:^|\s)src\s*=\s*"([^"]+)"/i) {
      my $src = $1;
      my $data = to_data_uri($base, $src);
      if (defined $data) {
        $tag =~ s/(?:^|\s)src\s*=\s*"[^"]+"/ src="$data"/i;
        $tag =~ s/\sdata-src\s*=\s*"[^"]*"//ig;
        $tag =~ s/\sdata-src\s*=\s*'\''[^'\'']*'\''//ig;
      }
    }
    $tag;
  }geisx;

  # Also inline CSS/background image references like url("images/xxx.png").
  $html =~ s{
    url\((["\x27]?)(images/[^)"\x27\s]+)\1\)
  }{
    my ($q, $src) = ($1, $2);
    my $data = to_data_uri($base, $src);
    defined $data ? "url(\"$data\")" : $&;
  }geisx;

  open my $of, ">:raw", $out_file or die "write html failed: $!";
  print {$of} $html;
  close $of;
' "$STATIC_HTML" "$EMBEDDED_HTML"

echo "[7/8] Converting to Markdown..."
if command -v pandoc >/dev/null 2>&1; then
  pandoc -f html -t gfm --wrap=none "$STATIC_HTML" -o "$MD_FILE" || true
else
  echo "WARN: pandoc not found, skip markdown conversion."
fi

echo "[8/8] Writing process record..."
IMG_TOTAL="$(find "$IMG_DIR" -type f | wc -l | tr -d ' ')"
IMG_IN_STATIC="$(
  perl -0777 -ne '
    my $n=0;
    while (/<img\b[^>]*\bsrc\s*=\s*["\x27]images\/[^"\x27]+["\x27][^>]*>/ig) { $n++ }
    print $n;
  ' "$STATIC_HTML"
)"
IMG_IN_EMBEDDED="$(
  perl -0777 -ne '
    my $n=0;
    while (/<img\b[^>]*\bsrc\s*=\s*["\x27]data:image\/[^"\x27]+["\x27][^>]*>/ig) { $n++ }
    print $n;
  ' "$EMBEDDED_HTML"
)"

cat > "$PROCESS_MD" <<EOF
# WeChat 导出流程记录

- 运行时间: $(date '+%Y-%m-%d %H:%M:%S %z')
- 原始链接: $URL
- 输出目录: $OUT_DIR

## 流程
1. 使用移动端微信 UA 抓取原始网页，保存为 \`raw_page.html\`。
2. 从原始网页中解析并提取 \`div#js_content\` 内容，生成 \`js_content.html\`。
3. 提取 \`js_content\` 中所有 \`mmbiz.qpic.cn\` 图片资源 URL，去重后下载到 \`images/\`。
4. 生成 URL 到本地文件映射 \`image_map.tsv\`。
5. 将文章 HTML 中图片地址替换为本地路径，并强制写入 \`src\`，生成 \`article.static.html\`。
6. 将本地图片进一步内嵌为 base64 data URI，生成 \`article.embedded.html\`。
7. （可选）使用 pandoc 将 \`article.static.html\` 转成 \`article.md\`。

## 产物
- \`raw_page.html\`
- \`js_content.inner.html\`
- \`js_content.html\`
- \`image_urls.txt\`
- \`image_map.tsv\`
- \`images/\` (下载图片总数: $IMG_TOTAL)
- \`article.static.html\` (本地图片 src 数: $IMG_IN_STATIC)
- \`article.embedded.html\` (内嵌图片 src 数: $IMG_IN_EMBEDDED)
- \`article.md\`（若已安装 pandoc）
- \`run.log\`

## 打开方式
- 推荐离线查看: \`article.embedded.html\`（单文件，不依赖外部图片目录）
- 保持相对目录查看: \`article.static.html\`（需与 \`images/\` 同目录）
EOF

echo
echo "Done."
echo "Output directory: $OUT_DIR"
echo "Recommended file: $EMBEDDED_HTML"
