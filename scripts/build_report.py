"""장별 Markdown 리포트를 단일 HTML 로 병합한다.

    python scripts/build_report.py            # config.report.output_html 생성
    python scripts/build_report.py --open     # 생성 후 경로만 출력

`config.report.source_dir` 의 `*.md` 를 **파일명 순서**로 이어 붙이고, A4 인쇄용 CSS(`@page`,
페이지 번호, 페이지 나눔)와 목차를 넣는다. 브라우저에서 "PDF로 저장"하면 제출용 PDF 가 된다.

의존성을 늘리지 않으려고 Markdown 변환기를 직접 구현했다. 이 리포트가 쓰는 문법만 다룬다:
제목, 문단, 강조(`**`), 인라인 코드, 링크, 이미지, 표(정렬 지정 포함), 순서·비순서 목록,
인용문, 코드 블록(```), 수평선, `<!-- TOC -->` 자리표시자.

이미지·문서 링크의 `../` 는 출력 HTML 위치(`docs/`) 기준으로 다시 쓴다.
한글 폰트는 시스템 폰트로 대체(fallback)한다. 웹 폰트를 받지 않으므로 오프라인에서도 동일하게 열린다.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path  # noqa: E402

CSS = """
:root { --ink:#1a1a1a; --muted:#5b6166; --line:#d8dce0; --accent:#1f4e79; --band:#f2f5f8; }

@page {
  size: A4;
  margin: 18mm 16mm 20mm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 8.5pt; color: #7a7f85; }
}
@page :first { @bottom-center { content: none; } }

* { box-sizing: border-box; }
body {
  font-family: "Pretendard", "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic",
               "NanumGothic", "Nanum Gothic", "AppleGothic", sans-serif;
  font-size: 10pt; line-height: 1.62; color: var(--ink);
  margin: 0 auto; padding: 0 12mm; max-width: 210mm;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
  word-break: keep-all; overflow-wrap: anywhere;
}
section.chapter { break-before: page; }
section.cover, section.toc { break-after: page; }
section.cover { break-before: auto; padding-top: 28mm; text-align: center; }
section.cover h1 { font-size: 26pt; border: 0; margin-bottom: 2mm; }
section.cover h2 { font-size: 14pt; color: var(--muted); border: 0; font-weight: 500; }
section.cover table { margin: 12mm auto; width: 88%; }
section.cover blockquote { text-align: left; margin-top: 14mm; }

h1, h2, h3, h4 { break-after: avoid; break-inside: avoid; line-height: 1.35; }
h1 { font-size: 19pt; margin: 0 0 6mm; padding-bottom: 2.5mm; border-bottom: 2.2pt solid var(--accent); }
h2 { font-size: 13.5pt; margin: 8mm 0 3mm; color: var(--accent); }
h3 { font-size: 11pt; margin: 6mm 0 2mm; }
h4 { font-size: 10pt; margin: 5mm 0 2mm; color: var(--muted); }
p { margin: 0 0 3mm; }
p.note { font-size: 8.6pt; color: var(--muted); margin-top: -1.5mm; }
strong { font-weight: 700; }
a { color: var(--accent); text-decoration: none; }
code { font-family: "D2Coding", "Consolas", "Nanum Gothic Coding", "DejaVu Sans Mono", monospace;
       font-size: 0.88em; background: var(--band); padding: 0.5pt 2pt; border-radius: 2pt; }
pre { background: var(--band); border: 0.6pt solid var(--line); border-radius: 3pt;
      padding: 3mm 4mm; overflow: hidden; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.6pt; line-height: 1.35; white-space: pre; }
blockquote { margin: 4mm 0; padding: 2.5mm 4mm; border-left: 2.5pt solid var(--accent);
             background: var(--band); break-inside: avoid; }
blockquote p:last-child { margin-bottom: 0; }
hr { border: 0; border-top: 0.6pt solid var(--line); margin: 6mm 0; }

table { width: 100%; border-collapse: collapse; margin: 3mm 0 4mm; font-size: 8.4pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 0.5pt solid var(--line); padding: 1.4mm 2mm; vertical-align: top; }
th { background: var(--band); font-weight: 700; text-align: left; }

figure { margin: 4mm 0 5mm; text-align: center; break-inside: avoid; }
figure img { display: block; margin: 0 auto; max-width: 100%; max-height: 112mm; }
figcaption { font-size: 8.4pt; color: var(--muted); margin-top: 1.5mm; }

ul, ol { margin: 0 0 3mm; padding-left: 7mm; }
li { margin-bottom: 1.2mm; }

section.toc h1 { border-bottom-width: 1.2pt; }
nav.toc ol { list-style: none; padding-left: 0; counter-reset: none; }
nav.toc > ol > li { margin: 2mm 0 1mm; font-weight: 700; }
nav.toc ol ol { padding-left: 6mm; margin: 0.8mm 0; }
nav.toc ol ol li { font-weight: 400; font-size: 9pt; color: var(--muted); }
"""

FENCE = re.compile(r"^```")
HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
IMAGE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
TABLE_SEP = re.compile(r"^\|[\s:|-]+\|\s*$")
UL_ITEM = re.compile(r"^([-*])\s+(.*)$")
OL_ITEM = re.compile(r"^(\d+)\.\s+(.*)$")
NOTE = re.compile(r"^(그림\s*)?각주[:：]")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_SPAN = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*(.+?)\*\*")


@dataclass
class Heading:
    level: int
    text: str
    anchor: str


@dataclass
class Document:
    body: str
    headings: list[Heading] = field(default_factory=list)


def rewrite(url: str) -> str:
    """장 파일(docs/report/) 기준 상대경로를 출력 HTML(docs/) 기준으로 바꾼다."""
    return url[3:] if url.startswith("../") else url


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = CODE_SPAN.sub(stash, text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    text = LINK.sub(lambda m: f'<a href="{html.escape(rewrite(m.group(2)), quote=True)}">{m.group(1)}</a>', text)
    text = text.replace("\\|", "|").replace("\\*", "*")
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{spans[int(m.group(1))]}</code>", text)


def split_row(line: str) -> list[str]:
    """표 한 줄을 셀로 나눈다. 이스케이프된 파이프(\\|)는 나누지 않는다."""
    cells, current, escaped = [], "", False
    for char in line.strip().strip("|"):
        if escaped:
            current += "\\" + char
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append(current)
            current = ""
        else:
            current += char
    cells.append(current)
    return [c.strip() for c in cells]


def alignments(separator: str) -> list[str]:
    out = []
    for cell in split_row(separator):
        left, right = cell.startswith(":"), cell.endswith(":")
        out.append("center" if left and right else "right" if right else "left")
    return out


def convert(lines: list[str], doc: Document, slug: str) -> str:
    html_parts: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "<!-- TOC -->":
            html_parts.append("\x01TOC\x01")
            i += 1
            continue

        if FENCE.match(stripped):
            i += 1
            block = []
            while i < len(lines) and not FENCE.match(lines[i].strip()):
                block.append(lines[i])
                i += 1
            i += 1
            html_parts.append(f"<pre><code>{html.escape(chr(10).join(block), quote=False)}</code></pre>")
            continue

        heading = HEADING.match(stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            anchor = f"{slug}-h{len(doc.headings)}"
            doc.headings.append(Heading(level, text, anchor))
            html_parts.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            i += 1
            continue

        image = IMAGE.match(stripped)
        if image:
            alt, src = image.group(1), rewrite(image.group(2))
            caption = f"<figcaption>{inline(alt)}</figcaption>" if alt else ""
            html_parts.append(
                f'<figure><img src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}">{caption}</figure>'
            )
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and TABLE_SEP.match(lines[i + 1].strip()):
            header = split_row(stripped)
            align = alignments(lines[i + 1])
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i].strip()))
                i += 1
            def cell(tag: str, value: str, index: int) -> str:
                style = f' style="text-align:{align[index]}"' if index < len(align) and align[index] != "left" else ""
                return f"<{tag}{style}>{inline(value)}</{tag}>"
            head = "".join(cell("th", c, n) for n, c in enumerate(header))
            body = "".join("<tr>" + "".join(cell("td", c, n) for n, c in enumerate(row)) + "</tr>" for row in rows)
            html_parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue

        if stripped.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            html_parts.append(f"<blockquote>{convert(quote, doc, slug)}</blockquote>")
            continue

        if stripped in ("---", "***", "___"):
            html_parts.append("<hr>")
            i += 1
            continue

        item = UL_ITEM.match(stripped) or OL_ITEM.match(stripped)
        if item:
            ordered = bool(OL_ITEM.match(stripped))
            items: list[list[str]] = []
            while i < len(lines):
                current = lines[i]
                match = OL_ITEM.match(current.strip()) if ordered else UL_ITEM.match(current.strip())
                if match and not current.startswith("   "):
                    items.append([match.group(2)])
                elif current.strip() and items and current.startswith(" "):
                    items[-1].append(current.strip())        # 이어지는 줄
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            rendered = "".join(f"<li>{inline(' '.join(parts))}</li>" for parts in items)
            html_parts.append(f"<{tag}>{rendered}</{tag}>")
            continue

        paragraph = []
        while i < len(lines) and lines[i].strip() and not _is_block_start(lines, i):
            paragraph.append(lines[i].strip())
            i += 1
        text = " ".join(paragraph)
        css_class = ' class="note"' if NOTE.match(text) else ""
        html_parts.append(f"<p{css_class}>{inline(text)}</p>")

    return "\n".join(html_parts)


def _is_block_start(lines: list[str], i: int) -> bool:
    stripped = lines[i].strip()
    if i == 0:
        return False
    return bool(
        FENCE.match(stripped) or HEADING.match(stripped) or IMAGE.match(stripped)
        or stripped.startswith(">") or stripped == "<!-- TOC -->"
        or stripped in ("---", "***", "___")
        or UL_ITEM.match(stripped) or OL_ITEM.match(stripped)
        or (stripped.startswith("|") and i + 1 < len(lines) and TABLE_SEP.match(lines[i + 1].strip()))
    )


def build_toc(sections: list[tuple[str, Document]]) -> str:
    out = ['<nav class="toc"><ol>']
    for slug, doc in sections:
        if slug == "00_cover":
            continue
        for heading in doc.headings:
            if heading.level == 1:
                out.append(f'<li><a href="#{heading.anchor}">{inline(heading.text)}</a>')
                children = [h for h in doc.headings if h.level == 2]
                if children:
                    out.append("<ol>")
                    out += [f'<li><a href="#{h.anchor}">{inline(h.text)}</a></li>' for h in children]
                    out.append("</ol>")
                out.append("</li>")
    out.append("</ol></nav>")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--open", action="store_true", help="생성 후 파일 경로만 출력한다")
    args = parser.parse_args()

    cfg = load_config()["report"]
    source = resolve_path(cfg["source_dir"])
    output = resolve_path(cfg["output_html"])
    files = sorted(source.glob("*.md"))
    if not files:
        print(f"장 파일이 없습니다: {source}", file=sys.stderr)
        return 1

    sections: list[tuple[str, Document]] = []
    for path in files:
        doc = Document(body="")
        doc.body = convert(path.read_text(encoding="utf-8").splitlines(), doc, path.stem)
        sections.append((path.stem, doc))

    toc = build_toc(sections)
    parts = []
    for slug, doc in sections:
        if slug == "00_cover":
            body = doc.body.replace("\x01TOC\x01", "")
            parts.append(f'<section class="cover" id="{slug}">{body}</section>')
            parts.append(f'<section class="toc" id="toc"><h1>목차</h1>{toc}</section>')
        else:
            parts.append(f'<section class="chapter" id="{slug}">{doc.body}</section>')

    title = cfg["title"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "<!doctype html>\n<html lang=\"ko\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        + "\n".join(parts)
        + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    chapters = sum(1 for slug, _ in sections if slug not in ("00_cover",))
    figures = sum(doc.body.count("<figure>") for _, doc in sections)
    tables = sum(doc.body.count("<table>") for _, doc in sections)
    size_kb = output.stat().st_size / 1024
    if args.open:
        print(output)
        return 0
    print(f"생성: {output} ({size_kb:,.0f} KB)")
    print(f"  장 {chapters}개 · 그림 {figures}개 · 표 {tables}개")
    print("  브라우저에서 열고 '인쇄 → PDF로 저장'(용지 A4, 배경 그래픽 켜기, 머리글/바닥글 끄기)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
