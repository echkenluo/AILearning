import re, sys
from html.parser import HTMLParser
from html import unescape

OUTDIR = sys.argv[1]

class WxToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
        self.in_pre = False
        self.list_depth = 0
        self.link_stack = []
        self.heading_as_bold = False

    def _out(self, text):
        self.parts.append(text)

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ('script', 'style', 'svg'):
            self.skip += 1; return
        if self.skip:
            return
        # LaTeX formula
        formula = d.get('data-formula')
        if formula:
            formula = unescape(formula).strip()
            if '\n' in formula or len(formula) > 80:
                self._out(f'\n\n$$\n{formula}\n$$\n\n')
            else:
                self._out(f' ${formula}$ ')
            self.skip += 1; return
        # Image
        if tag == 'img':
            src = d.get('src', d.get('data-src', ''))
            if 'mmbiz.qpic.cn' in src:
                local = url_map.get(src)
                if not local:
                    base = src.split('?')[0]
                    for k, v in url_map.items():
                        if k.split('?')[0] == base:
                            local = v; break
                if local:
                    self._out(f'\n\n![]({local})\n\n')
            return
        # Headings
        if tag in ('h1','h2','h3','h4','h5','h6'):
            level = int(tag[1])
            if level >= 5:
                # h5/h6 render too small; use bold text instead
                self._out('\n\n**')
                self.heading_as_bold = True
            else:
                self._out('\n\n' + '#' * level + ' ')
                self.heading_as_bold = False
        elif tag == 'p':
            self._out('\n\n')
        elif tag == 'br':
            self._out('\n')
        elif tag in ('strong', 'b'):
            self._out('**')
        elif tag in ('em', 'i'):
            self._out('*')
        elif tag == 'code' and not self.in_pre:
            self._out('`')
        elif tag == 'pre':
            self.in_pre = True; self._out('\n\n```\n')
        elif tag in ('ul', 'ol'):
            self.list_depth += 1; self._out('\n')
        elif tag == 'li':
            indent = '  ' * max(0, self.list_depth - 1)
            self._out(f'\n{indent}- ')
        elif tag == 'blockquote':
            self._out('\n\n> ')
        elif tag == 'a':
            href = d.get('href', '')
            if href and not href.startswith('javascript'):
                self._out('['); self.link_stack.append(href)
        elif tag == 'sup':
            self._out('[^')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'svg'):
            self.skip = max(0, self.skip - 1); return
        if self.skip:
            if tag == 'span': self.skip = max(0, self.skip - 1)
            return
        if tag in ('h1','h2','h3','h4','h5','h6'):
            if self.heading_as_bold:
                self._out('**\n\n')
                self.heading_as_bold = False
            else:
                self._out('\n\n')
        elif tag in ('strong', 'b'):
            self._out('**')
        elif tag in ('em', 'i'):
            self._out('*')
        elif tag == 'code' and not self.in_pre:
            self._out('`')
        elif tag == 'pre':
            self.in_pre = False; self._out('\n```\n\n')
        elif tag in ('ul', 'ol'):
            self.list_depth = max(0, self.list_depth - 1); self._out('\n')
        elif tag == 'blockquote':
            self._out('\n\n')
        elif tag == 'a' and self.link_stack:
            self._out(f']({self.link_stack.pop()})')
        elif tag == 'sup':
            self._out(']')

    def handle_data(self, data):
        if self.skip: return
        if self.in_pre: self._out(data); return
        if not data.strip():
            if data and self.parts and not self.parts[-1].endswith('\n'):
                self._out(' ')
            return
        self._out(data)

    def handle_entityref(self, name):
        if self.skip: return
        self._out({'nbsp':' ','lt':'<','gt':'>','amp':'&','quot':'"'}.get(name, f'&{name};'))

    def handle_charref(self, name):
        if self.skip: return
        self._out(chr(int(name)))

    def get_result(self):
        md = ''.join(self.parts)
        # Fix empty list items: lone "- " followed by blank lines then content
        md = re.sub(r'\n-\s*\n\n+', '\n\n', md)
        # Clean excessive blank lines
        md = re.sub(r'\n{4,}', '\n\n\n', md)
        md = re.sub(r'[ \t]+\n', '\n', md)
        # Fix reference section: "参考资料[^1]" -> "参考资料\n\n[^1]"
        md = re.sub(r'(参考资料)\s*(\[)', r'\1\n\n\2', md)
        # Split concatenated references: "*url*[^2]" -> "*url*\n\n[^2]"
        md = re.sub(r'\*\s*(\[\^)', r'*\n\n\1', md)
        return md.strip() + '\n'

# Build URL -> local path mapping
url_map = {}
import os
tsv_path = f'{OUTDIR}/image_map.tsv'
txt_path = f'{OUTDIR}/img_urls.txt'
if os.path.exists(tsv_path):
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                url, local = line.split('\t', 1)
                url_map[url] = local
else:
    with open(txt_path) as f:
        for i, line in enumerate(f, 1):
            url_map[line.strip()] = f'images/img_{i:02d}.png'

# Extract title from raw HTML
title = 'Article'
with open(f'{OUTDIR}/raw.html') as f:
    raw = f.read()
    m = re.search(r'var\s+msg_title\s*=\s*["\']([^"\']+)', raw)
    if m: title = unescape(m.group(1))

# Parse content HTML
with open(f'{OUTDIR}/content.html') as f:
    html = f.read()

parser = WxToMarkdown()
parser.feed(html)
md = f'# {title}\n\n' + parser.get_result()

# Post-processing fixes
md = re.sub(r'\[\^\[(\d+)\]\]', r'[^\1]', md)           # [^[1]] -> [^1]
md = re.sub(r'\*\s*\[(\d+)\]\s*\n', r'*\n\n[\1]\n', md)  # Split refs
md = re.sub(r'(\S)参考资料', r'\1\n\n---\n\n## 参考资料', md)
md = md.replace('## ## 参考资料', '## 参考资料')
md = re.sub(r'\\lt(?=[ }])', '<', md)                         # \lt -> <

# Fix $$ blocks inside list context: append as inline math to preceding text
# VSCode Officeviewer can't render $$ blocks in lists, and standalone $...$
# lines are not recognized as math either — must be inline within text.
lines = md.split('\n')
fixed = []
i = 0
while i < len(lines):
    if lines[i].strip() == '$$':
        # Check if we're in a list context
        j = len(fixed) - 1
        while j >= 0 and fixed[j].strip() == '':
            j -= 1
        prev = fixed[j].strip() if j >= 0 else ''
        in_list = prev.startswith('- ') or prev.startswith('  ')
        if in_list:
            # Collect the formula content between $$ ... $$
            formula_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                formula_lines.append(lines[i])
                i += 1
            formula = ' '.join(l.strip() for l in formula_lines).strip()
            # Remove trailing blank lines and append formula to prev text line
            while fixed and fixed[-1].strip() == '':
                fixed.pop()
            fixed[-1] = fixed[-1].rstrip() + f' $\\displaystyle {formula}$'
            i += 1  # skip closing $$
            continue
    fixed.append(lines[i])
    i += 1
md = '\n'.join(fixed)

with open(f'{OUTDIR}/article.md', 'w') as f:
    f.write(md)

img_count = len(re.findall(r'!\[.*?\]\(images/', md))
formula_count = len(re.findall(r'\$[^$\n]+\$', md))
print(f'  Output: {OUTDIR}/article.md')
print(f'  Size: {len(md)} bytes, {md.count(chr(10))} lines')
print(f'  Images: {img_count}')
print(f'  Formulas: ~{formula_count}')
