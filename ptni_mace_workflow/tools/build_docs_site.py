#!/usr/bin/env python
"""Build a static HTML documentation site from the refactored PtNi MACE docs.

The output is a self-contained static site under
``mace_workspace/reports/docs_site``. It can be opened directly or deployed with
any static web server.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DOCS = [
    {
        "id": "overview",
        "source": "ptni_mace_workflow/docs/README_中文.md",
        "title": "PtNi MACE 重构总览",
        "subtitle": "模块边界、workspace 目录和常用入口",
    },
    {
        "id": "preprocess",
        "source": "ptni_mace_workflow/docs/01_前处理.md",
        "title": "前处理",
        "subtitle": "OUTCAR 清洗、POTCAR 筛选、extxyz 转换和数据集切分",
    },
    {
        "id": "training",
        "source": "ptni_mace_workflow/docs/02_训练.md",
        "title": "训练",
        "subtitle": "fine-tune、scratch、checkpoint 保存、best-loss model 导出",
    },
    {
        "id": "evaluation",
        "source": "ptni_mace_workflow/docs/03_误差检验.md",
        "title": "误差检验",
        "subtitle": "低显存预测、train/valid/test 打分、离群点和 parity 图",
    },
    {
        "id": "benchmarks",
        "source": "ptni_mace_workflow/docs/04_外推任务验证.md",
        "title": "外推任务验证",
        "subtitle": "晶格、应变 NEB、Pt111 PES、距离稳定性、NP 单点和 NP relax+NEB",
    },
    {
        "id": "workspace",
        "source": "ptni_mace_workflow/docs/05_workspace目录规范.md",
        "title": "workspace 目录规范",
        "subtitle": "数据、模型、运行结果和报告的隔离约定",
    },
    {
        "id": "migration",
        "source": "ptni_mace_workflow/docs/06_从旧outputs迁移说明.md",
        "title": "从旧 outputs 迁移",
        "subtitle": "旧脚本保留、规范数据迁移和新旧路径对照",
    },
    {
        "id": "git-maintenance",
        "source": "ptni_mace_workflow/docs/07_Git与GitHub维护教程.md",
        "title": "Git 与 GitHub 维护教程",
        "subtitle": "面向新手的提交、版本号、tag、push 和 Pages 发布流程",
    },
    {
        "id": "mcmd",
        "source": "ptni_mace_workflow/docs/08_MCMD框架.md",
        "title": "semi-rfKMC / MCMD 动力学雏形",
        "subtitle": "随机低配位原子、局部全 He 候选 CI-NEB 和速率概率选择",
    },
]


@dataclass
class TocItem:
    level: int
    title: str
    anchor: str


@dataclass
class RenderedDoc:
    doc_id: str
    title: str
    subtitle: str
    source_name: str
    source_path: Path
    updated_at: str
    html: str
    toc: list[TocItem]
    text: str


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)

    def replace_link(match: re.Match) -> str:
        label = match.group(1)
        href = match.group(2)
        safe_href = html.escape(href, quote=True)
        if href.startswith(("http://", "https://", "#", "./", "../")) or not re.match(r"^[A-Za-z]:", href):
            return f'<a href="{safe_href}">{label}</a>'
        return f"<code>{label}</code>"

    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def close_blocks(parts: list[str], list_stack: list[str], paragraph: list[str]) -> None:
    if paragraph:
        parts.append(f"<p>{inline_markdown(' '.join(paragraph))}</p>")
        paragraph.clear()
    while list_stack:
        parts.append(f"</{list_stack.pop()}>")


def render_table(lines: list[str]) -> str:
    header = split_table_row(lines[0])
    rows = [split_table_row(line) for line in lines[2:]]
    out = ["<div class=\"table-wrap\"><table>"]
    out.append("<thead><tr>")
    for cell in header:
        out.append(f"<th>{inline_markdown(cell)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{inline_markdown(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def render_markdown(source: str, doc_id: str) -> tuple[str, list[TocItem], str]:
    lines = source.splitlines()
    parts: list[str] = []
    toc: list[TocItem] = []
    text_chunks: list[str] = []
    paragraph: list[str] = []
    list_stack: list[str] = []
    heading_count = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            close_blocks(parts, list_stack, paragraph)
            lang = stripped[3:].strip() or "text"
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code_text = "\n".join(code_lines)
            text_chunks.append(code_text)
            parts.append(
                "<pre class=\"code-block\" data-language=\"{}\"><code>{}</code></pre>".format(
                    html.escape(lang, quote=True),
                    html.escape(code_text),
                )
            )
            i += 1
            continue

        if not stripped:
            close_blocks(parts, list_stack, paragraph)
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            close_blocks(parts, list_stack, paragraph)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_count += 1
            anchor = f"{doc_id}-h{heading_count}"
            toc.append(TocItem(level=level, title=re.sub(r"`", "", title), anchor=anchor))
            text_chunks.append(title)
            parts.append(
                f'<h{level} id="{anchor}" data-anchor="{anchor}">'
                f"{inline_markdown(title)}"
                f'<a class="anchor-link" href="#{anchor}" aria-label="链接到本节">#</a>'
                f"</h{level}>"
            )
            i += 1
            continue

        if i + 1 < len(lines) and stripped.startswith("|") and is_table_separator(lines[i + 1]):
            close_blocks(parts, list_stack, paragraph)
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            text_chunks.extend(table_lines)
            parts.append(render_table(table_lines))
            continue

        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if unordered or ordered:
            if paragraph:
                parts.append(f"<p>{inline_markdown(' '.join(paragraph))}</p>")
                paragraph.clear()
            tag = "ul" if unordered else "ol"
            if not list_stack or list_stack[-1] != tag:
                while list_stack:
                    parts.append(f"</{list_stack.pop()}>")
                parts.append(f"<{tag}>")
                list_stack.append(tag)
            item_text = (unordered or ordered).group(1)
            checkbox = re.match(r"\[(x|X| )\]\s+(.+)$", item_text)
            if checkbox:
                checked = " checked" if checkbox.group(1).lower() == "x" else ""
                item_body = checkbox.group(2)
                parts.append(
                    f'<li class="task-item"><input type="checkbox" disabled{checked}> '
                    f"{inline_markdown(item_body)}</li>"
                )
            else:
                parts.append(f"<li>{inline_markdown(item_text)}</li>")
            text_chunks.append(item_text)
            i += 1
            continue

        if stripped.startswith(">"):
            close_blocks(parts, list_stack, paragraph)
            quote = stripped.lstrip(">").strip()
            parts.append(f"<blockquote>{inline_markdown(quote)}</blockquote>")
            text_chunks.append(quote)
            i += 1
            continue

        paragraph.append(stripped)
        text_chunks.append(stripped)
        i += 1

    close_blocks(parts, list_stack, paragraph)
    return "\n".join(parts), toc, "\n".join(text_chunks)


def render_doc(doc: dict, workspace: Path) -> RenderedDoc:
    source_path = workspace / doc["source"]
    source = source_path.read_text(encoding="utf-8")
    rendered, toc, plain_text = render_markdown(source, doc["id"])
    mtime = datetime.fromtimestamp(source_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return RenderedDoc(
        doc_id=doc["id"],
        title=doc["title"],
        subtitle=doc["subtitle"],
        source_name=source_path.name,
        source_path=source_path,
        updated_at=mtime,
        html=rendered,
        toc=toc,
        text=plain_text,
    )


def toc_html(doc: RenderedDoc) -> str:
    items = []
    for item in doc.toc:
        if item.level > 3:
            continue
        items.append(
            '<a class="toc-link level-{level}" href="#{anchor}" data-doc="{doc_id}" '
            'data-target="{anchor}">{title}</a>'.format(
                level=item.level,
                anchor=item.anchor,
                doc_id=doc.doc_id,
                title=html.escape(item.title),
            )
        )
    return "\n".join(items)


def article_html(doc: RenderedDoc, active: bool) -> str:
    active_class = " active" if active else ""
    return f"""
<article class="doc-page{active_class}" id="doc-{doc.doc_id}" data-doc="{doc.doc_id}">
  <div class="doc-heading">
    <div>
      <p class="eyebrow">在线文档</p>
      <h1>{html.escape(doc.title)}</h1>
      <p>{html.escape(doc.subtitle)}</p>
    </div>
    <div class="doc-actions">
      <a class="button" href="sources/{html.escape(doc.source_name)}" target="_blank" rel="noreferrer">查看 Markdown</a>
      <button class="button ghost" type="button" data-print>打印</button>
    </div>
  </div>
  <div class="doc-meta">
    <span>源文件: {html.escape(doc.source_name)}</span>
    <span>更新: {html.escape(doc.updated_at)}</span>
    <span>章节: {len(doc.toc)}</span>
  </div>
  <div class="content">
    {doc.html}
  </div>
</article>
"""


def build_index(rendered_docs: list[RenderedDoc]) -> str:
    tabs = []
    toc_blocks = []
    articles = []
    search_index = []
    for index, doc in enumerate(rendered_docs):
        active = index == 0
        tabs.append(
            '<button class="doc-tab{active}" type="button" data-doc="{doc_id}">{title}</button>'.format(
                active=" active" if active else "",
                doc_id=doc.doc_id,
                title=html.escape(doc.title),
            )
        )
        toc_blocks.append(
            '<nav class="toc-panel{active}" data-doc="{doc_id}">{toc}</nav>'.format(
                active=" active" if active else "",
                doc_id=doc.doc_id,
                toc=toc_html(doc),
            )
        )
        articles.append(article_html(doc, active))
        for item in doc.toc:
            search_index.append(
                {
                    "doc": doc.doc_id,
                    "docTitle": doc.title,
                    "anchor": item.anchor,
                    "title": item.title,
                    "level": item.level,
                }
            )

    search_json = json.dumps(search_index, ensure_ascii=False, separators=(",", ":"))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>PtNi MACE 重构文档中心</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <span class="brand-mark">PtNi</span>
      <div>
        <strong>MACE 重构文档中心</strong>
        <span>前处理 / 训练 / 评估 / 外推验证</span>
      </div>
    </div>
    <div class="top-actions">
      <label class="search-box">
        <span>搜索</span>
        <input id="searchInput" type="search" placeholder="搜索章节、脚本、命令..." autocomplete="off">
      </label>
      <a class="button compact" href="README.md" target="_blank" rel="noreferrer">部署说明</a>
    </div>
  </header>

  <div class="layout">
    <aside class="sidebar">
      <div class="doc-tabs">
        {"".join(tabs)}
      </div>
      <div id="searchResults" class="search-results" hidden></div>
      <div class="toc-title">目录</div>
      {"".join(toc_blocks)}
      <div class="build-note">生成时间: {html.escape(generated_at)}</div>
    </aside>
    <main class="main">
      {"".join(articles)}
    </main>
  </div>
  <script id="searchIndex" type="application/json">{search_json}</script>
  <script src="assets/app.js"></script>
</body>
</html>
"""


STYLE_CSS = r"""
:root {
  --bg: #f7f8f5;
  --panel: #ffffff;
  --panel-alt: #eef4f1;
  --ink: #202522;
  --muted: #64706a;
  --line: #dbe2dc;
  --accent: #126f6a;
  --accent-2: #b56b19;
  --code-bg: #17211f;
  --code-ink: #edf7f4;
  --shadow: 0 14px 36px rgba(35, 45, 41, 0.10);
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font-family: "Inter", "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
  line-height: 1.65;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  min-height: 68px;
  padding: 10px 22px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(12px);
}

.brand { display: flex; align-items: center; gap: 12px; min-width: 260px; }
.brand-mark {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  color: #fff;
  background: var(--accent);
  border-radius: 8px;
  font-weight: 750;
}
.brand strong { display: block; font-size: 16px; }
.brand span:last-child { display: block; color: var(--muted); font-size: 12px; }

.top-actions { display: flex; align-items: center; gap: 12px; }
.search-box {
  display: flex;
  align-items: center;
  gap: 8px;
  width: min(42vw, 470px);
  padding: 7px 10px;
  background: var(--panel-alt);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.search-box span { color: var(--muted); font-size: 12px; }
.search-box input {
  width: 100%;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--ink);
  font-size: 14px;
}

.layout { display: grid; grid-template-columns: 320px minmax(0, 1fr); min-height: calc(100vh - 68px); }
.sidebar {
  position: sticky;
  top: 68px;
  height: calc(100vh - 68px);
  overflow: auto;
  padding: 18px 16px;
  border-right: 1px solid var(--line);
  background: #fbfcf9;
}
.main { min-width: 0; padding: 24px; }
.doc-page {
  display: none;
  width: min(1120px, 100%);
  margin: 0 auto 44px;
  padding: 30px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.doc-page.active { display: block; }

.doc-tabs { display: grid; gap: 8px; margin-bottom: 16px; }
.doc-tab {
  width: 100%;
  padding: 10px 12px;
  text-align: left;
  color: var(--ink);
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 8px;
  cursor: pointer;
  font-weight: 650;
}
.doc-tab.active {
  color: #fff;
  background: var(--accent);
  border-color: var(--accent);
}

.toc-title {
  margin: 18px 0 8px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 750;
  letter-spacing: 0;
}
.toc-panel { display: none; }
.toc-panel.active { display: grid; gap: 2px; }
.toc-link {
  display: block;
  padding: 6px 8px;
  color: var(--muted);
  text-decoration: none;
  border-left: 2px solid transparent;
  border-radius: 0 6px 6px 0;
  font-size: 13px;
}
.toc-link:hover, .toc-link.active {
  color: var(--accent);
  background: #e8f2ef;
  border-left-color: var(--accent);
}
.toc-link.level-2 { padding-left: 16px; }
.toc-link.level-3 { padding-left: 28px; font-size: 12px; }

.search-results {
  display: grid;
  gap: 8px;
  margin: 12px 0 14px;
  padding: 10px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
}
.search-results[hidden] { display: none; }
.search-result {
  padding: 8px;
  color: var(--ink);
  text-decoration: none;
  border-radius: 6px;
}
.search-result:hover { background: var(--panel-alt); }
.search-result small { display: block; color: var(--muted); }

.doc-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--line);
}
.doc-heading h1 { margin: 0; font-size: 30px; line-height: 1.25; letter-spacing: 0; }
.doc-heading p { margin: 8px 0 0; color: var(--muted); }
.eyebrow {
  margin: 0 0 6px !important;
  color: var(--accent-2) !important;
  font-weight: 750;
  font-size: 12px;
}
.doc-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 34px;
  padding: 7px 11px;
  color: #fff;
  background: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 7px;
  text-decoration: none;
  font-size: 13px;
  cursor: pointer;
}
.button.ghost, .button.compact {
  color: var(--accent);
  background: #fff;
}
.button.compact { min-height: 32px; }

.doc-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 14px 0 24px;
}
.doc-meta span {
  padding: 4px 8px;
  color: var(--muted);
  background: var(--panel-alt);
  border-radius: 6px;
  font-size: 12px;
}

.content h1, .content h2, .content h3, .content h4 {
  position: relative;
  scroll-margin-top: 92px;
  letter-spacing: 0;
}
.content h1 { margin: 28px 0 12px; font-size: 27px; }
.content h2 {
  margin: 32px 0 12px;
  padding-top: 8px;
  border-top: 1px solid var(--line);
  font-size: 22px;
}
.content h3 { margin: 26px 0 10px; font-size: 18px; }
.content h4 { margin: 20px 0 8px; font-size: 16px; }
.anchor-link {
  margin-left: 8px;
  color: var(--accent);
  opacity: 0;
  text-decoration: none;
  font-size: 0.8em;
}
h1:hover .anchor-link, h2:hover .anchor-link, h3:hover .anchor-link, h4:hover .anchor-link { opacity: 1; }

.content p { margin: 10px 0; }
.content a { color: var(--accent); }
.content code {
  padding: 2px 5px;
  color: #0f4f4b;
  background: #e8f2ef;
  border-radius: 5px;
  font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 0.92em;
}
.content pre code { padding: 0; color: inherit; background: transparent; border-radius: 0; }
.code-block {
  position: relative;
  overflow: auto;
  margin: 14px 0;
  padding: 18px;
  color: var(--code-ink);
  background: var(--code-bg);
  border-radius: 8px;
  font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 13px;
}
.code-copy {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 4px 8px;
  color: var(--code-ink);
  background: rgba(255,255,255,0.11);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
}

.table-wrap { overflow: auto; margin: 16px 0; border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { text-align: left; color: #20312d; background: #eaf1ed; font-weight: 750; }
tr:last-child td { border-bottom: 0; }
tbody tr:nth-child(even) { background: #fbfcfa; }

blockquote {
  margin: 14px 0;
  padding: 10px 14px;
  background: #fff7ec;
  border-left: 4px solid var(--accent-2);
  border-radius: 0 8px 8px 0;
}
.task-item { list-style: none; margin-left: -18px; }
.task-item input { margin-right: 8px; }
.build-note { margin-top: 20px; color: var(--muted); font-size: 12px; }

@media (max-width: 980px) {
  .topbar { position: static; align-items: flex-start; flex-direction: column; }
  .top-actions, .search-box { width: 100%; }
  .layout { display: block; }
  .sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
  .main { padding: 14px; }
  .doc-page { padding: 18px; }
  .doc-heading { display: block; }
  .doc-actions { justify-content: flex-start; margin-top: 12px; }
}
"""


APP_JS = r"""
const tabs = [...document.querySelectorAll('.doc-tab')];
const pages = [...document.querySelectorAll('.doc-page')];
const tocPanels = [...document.querySelectorAll('.toc-panel')];
const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');
const searchIndex = JSON.parse(document.getElementById('searchIndex').textContent);
const domSearchIndex = [];

pages.forEach(page => {
  const doc = page.dataset.doc;
  const docTitle = tabs.find(tab => tab.dataset.doc === doc)?.textContent || doc;
  let current = { title: docTitle, anchor: page.querySelector('.content h1, .content h2, .content h3')?.id || doc };
  page.querySelectorAll('.content h1, .content h2, .content h3, .content p, .content li, .content td, .content pre').forEach(node => {
    const isHeading = /^H[1-3]$/.test(node.tagName);
    if (isHeading && node.id) current = { title: node.innerText.replace('#', '').trim(), anchor: node.id };
    const text = node.innerText.replace(/\s+/g, ' ').trim();
    if (!text || text.length < 2) return;
    domSearchIndex.push({
      doc,
      docTitle,
      anchor: isHeading && node.id ? node.id : current.anchor,
      title: isHeading ? text.replace('#', '').trim() : current.title,
      context: isHeading ? '' : text.slice(0, 180)
    });
  });
});

function activeDoc() {
  return document.querySelector('.doc-page.active')?.dataset.doc || pages[0].dataset.doc;
}

function switchDoc(docId, target) {
  tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.doc === docId));
  pages.forEach(page => page.classList.toggle('active', page.dataset.doc === docId));
  tocPanels.forEach(panel => panel.classList.toggle('active', panel.dataset.doc === docId));
  if (target) {
    const el = document.getElementById(target);
    if (el) window.setTimeout(() => el.scrollIntoView({ block: 'start' }), 20);
  } else {
    window.scrollTo({ top: 0 });
  }
}

tabs.forEach(tab => tab.addEventListener('click', () => {
  history.replaceState(null, '', `#${tab.dataset.doc}`);
  switchDoc(tab.dataset.doc);
}));

document.querySelectorAll('.toc-link').forEach(link => {
  link.addEventListener('click', event => {
    event.preventDefault();
    history.replaceState(null, '', link.getAttribute('href'));
    switchDoc(link.dataset.doc, link.dataset.target);
  });
});

function hydrateFromHash() {
  const raw = decodeURIComponent(location.hash.replace(/^#/, ''));
  if (!raw) return;
  const target = document.getElementById(raw);
  if (target) {
    const docId = target.closest('.doc-page')?.dataset.doc;
    if (docId) switchDoc(docId, raw);
    return;
  }
  if (pages.some(page => page.dataset.doc === raw)) switchDoc(raw);
}
window.addEventListener('hashchange', hydrateFromHash);
hydrateFromHash();

document.querySelectorAll('pre.code-block').forEach(pre => {
  const button = document.createElement('button');
  button.className = 'code-copy';
  button.type = 'button';
  button.textContent = '复制';
  button.addEventListener('click', async () => {
    await navigator.clipboard.writeText(pre.querySelector('code')?.innerText || pre.innerText);
    button.textContent = '已复制';
    window.setTimeout(() => button.textContent = '复制', 1200);
  });
  pre.appendChild(button);
});

document.querySelectorAll('[data-print]').forEach(button => {
  button.addEventListener('click', () => window.print());
});

function renderSearch(query) {
  const q = query.trim().toLowerCase();
  if (!q) {
    searchResults.hidden = true;
    searchResults.innerHTML = '';
    return;
  }
  const hits = searchIndex
    .concat(domSearchIndex)
    .filter(item => (item.title + ' ' + item.docTitle + ' ' + (item.context || '')).toLowerCase().includes(q))
    .slice(0, 18);
  searchResults.hidden = false;
  searchResults.innerHTML = hits.length
    ? hits.map(item => `<a class="search-result" href="#${item.anchor}" data-doc="${item.doc}" data-target="${item.anchor}">${item.title}<small>${item.docTitle}${item.context ? ' - ' + item.context : ''}</small></a>`).join('')
    : '<div class="search-result">没有匹配的章节</div>';
  searchResults.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      history.replaceState(null, '', link.getAttribute('href'));
      switchDoc(link.dataset.doc, link.dataset.target);
    });
  });
}
searchInput.addEventListener('input', event => renderSearch(event.target.value));
document.addEventListener('keydown', event => {
  if (event.key === '/' && document.activeElement !== searchInput) {
    event.preventDefault();
    searchInput.focus();
  }
});

const observer = new IntersectionObserver(entries => {
  const visible = entries
    .filter(entry => entry.isIntersecting)
    .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
  if (!visible) return;
  const id = visible.target.id;
  document.querySelectorAll('.toc-link').forEach(link => {
    link.classList.toggle('active', link.dataset.target === id);
  });
}, { rootMargin: '-90px 0px -70% 0px', threshold: 0.1 });

document.querySelectorAll('.content h1, .content h2, .content h3').forEach(heading => observer.observe(heading));
"""


README = """# PtNi MACE 文档站点

这是由 `ptni_mace_workflow/tools/build_docs_site.py` 生成的静态网页说明文档。

## 本地预览

从项目根目录运行：

```bash
python -m http.server 8088 --directory mace_workspace/reports/docs_site --bind 127.0.0.1
```

然后打开：

```text
http://127.0.0.1:8088
```

也可以直接打开 `index.html`。

## 重新生成

更新 Markdown 后运行：

```bash
python ptni_mace_workflow/tools/build_docs_site.py
```

## 静态部署

将整个 `mace_workspace/reports/docs_site/` 目录上传到任意静态网页服务即可，例如 GitHub Pages、Nginx、Apache、Cloudflare Pages 或服务器上的静态文件目录。
"""


def write_site(rendered_docs: list[RenderedDoc], out_dir: Path) -> None:
    assets_dir = out_dir / "assets"
    sources_dir = out_dir / "sources"
    assets_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "index.html").write_text(build_index(rendered_docs), encoding="utf-8")
    (assets_dir / "style.css").write_text(STYLE_CSS.strip() + "\n", encoding="utf-8")
    (assets_dir / "app.js").write_text(APP_JS.strip() + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(README, encoding="utf-8")
    for doc in rendered_docs:
        shutil.copy2(doc.source_path, sources_dir / doc.source_name)
    build_info = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "documents": [
            {
                "id": doc.doc_id,
                "title": doc.title,
                "source": str(doc.source_path),
                "updated_at": doc.updated_at,
                "toc_items": len(doc.toc),
            }
            for doc in rendered_docs
        ],
    }
    (out_dir / "build-info.json").write_text(
        json.dumps(build_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PtNi MACE static docs site.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mace_workspace/reports/docs_site"),
        help="Output directory. Default: mace_workspace/reports/docs_site.",
    )
    args = parser.parse_args()

    workspace = Path.cwd()
    rendered_docs = [render_doc(doc, workspace) for doc in DOCS]
    out_dir = args.out_dir.resolve()
    write_site(rendered_docs, out_dir)
    print(f"Docs site: {out_dir}")
    print(f"Index: {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
