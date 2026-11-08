#!/usr/bin/env python3
"""Build the companion website of "Computers in the Loop" from site.yml.

Writes plain HTML files (no JavaScript) to _site/:
    index.html          home page: the notes, the chapters, how to use the notebooks
    chNN/index.html     one page per chapter
    404.html            page-not-found page (GitHub Pages serves it)
    style.css, favicon.svg

By default it also copies the notebooks, scripts, figures and the book PDF into _site/, so
that a local preview has working downloads. The deploy workflow copies these itself and runs
this script with --no-copy.

Usage:
    python3 build_site.py                    # build into website/_site
    python3 build_site.py --out ../_site     # build somewhere else
    python3 build_site.py --book PATH.pdf    # use this PDF as the notes
    python3 build_site.py --no-copy          # pages only

Needs only the Python standard library. PyYAML is used when installed; otherwise a small
reader for the simple YAML of site.yml is used.
"""
import argparse
import html
import math
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent          # companion/website
ROOT = HERE.parent                              # companion (the repository root)


# --------------------------------------------------------------------------- YAML reading
def _scalar(text):
    """One YAML scalar: quoted string, integer, or plain string."""
    text = text.strip()
    if not text:
        return ""
    if text[0] == '"':
        end = text.rfind('"')
        body = text[1:end]
        return re.sub(r'\\(["\\nt])',
                      lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), body)
    if text[0] == "'":
        end = text.rfind("'")
        return text[1:end].replace("''", "'")
    text = re.sub(r"\s+#.*$", "", text)         # trailing comment on a plain scalar
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if text in ("true", "false"):
        return text == "true"
    return text


def _strip_comment(line):
    """Drop a comment that starts outside quotes."""
    quote_char = None
    for i, ch in enumerate(line):
        if quote_char:
            if ch == quote_char and line[i - 1] != "\\":
                quote_char = None
        elif ch in "\"'":
            quote_char = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def simple_yaml(text):
    """Read the subset of YAML used by site.yml: nested block mappings and lists,
    one-line scalars, comments. No flow style, anchors or multi-line strings."""
    lines = []
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if line.strip():
            lines.append((len(line) - len(line.lstrip(" ")), line.strip()))

    def block(i, indent):
        if lines[i][1].startswith("- "):
            return seq(i, indent)
        return mapping(i, indent)

    def mapping(i, indent):
        out = {}
        while i < len(lines) and lines[i][0] == indent and not lines[i][1].startswith("- "):
            key, _, rest = lines[i][1].partition(":")
            i += 1
            if rest.strip():
                out[key.strip()] = _scalar(rest)
            elif i < len(lines) and lines[i][0] > indent:
                out[key.strip()], i = block(i, lines[i][0])
            elif i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
                out[key.strip()], i = seq(i, indent)
            else:
                out[key.strip()] = None
        return out, i

    def seq(i, indent):
        out = []
        while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
            item = lines[i][1][2:]
            inner = indent + 2
            if re.match(r"^[\w-]+\s*:(\s|$)", item) and not item.startswith(('"', "'")):
                # a mapping that starts on the dash line: re-read it at the inner indent
                lines[i] = (inner, item)
                value, i = mapping(i, inner)
            else:
                value, i = _scalar(item), i + 1
            out.append(value)
        return out, i

    data, i = block(0, lines[0][0])
    if i != len(lines):
        raise ValueError(f"site.yml: could not read line: {lines[i][1]!r}")
    return data


def load_config(path):
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # noqa: F401
    except ImportError:
        return simple_yaml(text)
    return yaml.safe_load(text)


# --------------------------------------------------------------------------- helpers
def esc(s):
    return html.escape(str(s), quote=True)


def rich(s):
    """Escape, then print _x_ in italics (used for symbols such as T and z)."""
    return re.sub(r"(?<![\w])_([^_\s][^_]*?)_(?![\w])", r"<i>\1</i>", esc(s))


def plain(s):
    """Text without the italics markup (for <title> and meta tags)."""
    return re.sub(r"(?<![\w])_([^_\s][^_]*?)_(?![\w])", r"\1", str(s))


def mailto(address, subject, body=""):
    url = f"mailto:{address}?subject={quote(subject)}"
    if body:
        url += f"&body={quote(body)}"
    return url


def human_size(nbytes):
    return f"{nbytes / 1e6:.1f} MB" if nbytes >= 1e6 else f"{max(1, round(nbytes / 1e3))} kB"


def logo_svg(size, decorative=True):
    """The mark: a sine wave, its samples, and the zero-order-hold staircase."""
    x0, x1, period, amp, mid, step = 10, 54, 40, 15, 32, 8
    sine = " ".join(
        f"{x:.1f},{mid - amp * math.sin(2 * math.pi * (x - x0) / period):.1f}"
        for x in [x0 + k * (x1 - x0) / 44 for k in range(45)])
    samples = [(x, mid - amp * math.sin(2 * math.pi * (x - x0) / period))
               for x in range(x0, x1, step)]
    stairs = f"M{x0} {samples[0][1]:.1f}"
    for k, (x, y) in enumerate(samples):
        x_next = samples[k + 1][0] if k + 1 < len(samples) else x1
        stairs += f" V{y:.1f} H{x_next}"
    dots = "".join(f'<circle cx="{x}" cy="{y:.1f}" r="3"/>' for x, y in samples)
    a11y = 'aria-hidden="true" focusable="false"' if decorative else 'role="img" aria-label="Logo"'
    return (f'<svg class="mark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
            f'width="{size}" height="{size}" {a11y}>'
            f'<rect width="64" height="64" rx="14" fill="#1F2A33"/>'
            f'<polyline points="{sine}" fill="none" stroke="#7FC4CC" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="{stairs}" fill="none" stroke="#E67E22" stroke-width="4" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<g fill="#FFFFFF">{dots}</g></svg>')


# --------------------------------------------------------------------------- page frame
def page(cfg, *, title, description, prefix, body, home_href, canonical_path=""):
    """One HTML page. prefix leads from the page to the site root ('', '../', or the base path)."""
    site = plain(cfg["title"])
    full_title = esc(f"{plain(title)} · {site}" if title != cfg["title"] else
                     f"{site}: {plain(cfg['subtitle'])}")
    pages_url = f"https://{cfg['github_user']}.github.io{cfg['base_path']}"
    canonical = (f'\n<link rel="canonical" href="{esc(pages_url + canonical_path)}">'
                 if cfg["github_user"] != "USER" else "")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{full_title}</title>
<meta name="description" content="{esc(plain(description))}">
<meta name="author" content="{esc(cfg['author'])}">
<meta name="theme-color" content="#1F2A33">
<meta property="og:title" content="{full_title}">
<meta property="og:description" content="{esc(plain(description))}">
<meta property="og:type" content="website">{canonical}
<link rel="icon" href="{prefix}favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="{prefix}style.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site-header">
  <div class="wrap header-inner">
    <a class="brand" href="{home_href}">{logo_svg(36)}<span class="brand-text">{esc(site)}</span></a>
    <nav class="site-nav" aria-label="Site">
      <a href="{home_href}#chapters">Chapters</a>
      <a href="{home_href}#notebooks">Notebooks</a>
      <a href="{home_href}#corrections">Corrections</a>
    </nav>
  </div>
</header>
<main id="main">
{body}
</main>
<footer class="site-footer">
  <div class="wrap">
    <p>© {esc(cfg['author'])}</p>
    <p class="footer-sub">{esc(cfg['title'])}: {esc(plain(cfg['subtitle']))}</p>
  </div>
</footer>
</body>
</html>
"""


# --------------------------------------------------------------------------- home page
def chapter_card(ch, prefix):
    return f"""      <li class="card">
        <p class="card-num">Chapter {ch['number']}</p>
        <h4 class="card-title"><a href="{prefix}{ch['slug']}/">{rich(ch['title'])}</a></h4>
        <p class="card-text">{rich(ch['summary'])}</p>
      </li>"""


def home_page(cfg, book):
    chapters = cfg["chapters"]
    prefix = ""
    groups = []
    for part in cfg["parts"]:
        members = [c for c in chapters if part["first"] <= c["number"] <= part["last"]]
        span = (f"Chapter {part['first']}" if part["first"] == part["last"]
                else f"Chapters {part['first']}–{part['last']}")
        cards = "\n".join(chapter_card(c, prefix) for c in members)
        groups.append(f"""    <section class="part" aria-labelledby="part-{part['first']}">
      <h3 id="part-{part['first']}"><span class="part-span">{span}</span> {esc(part['name'])}</h3>
      <p class="part-summary">{rich(part['summary'])}</p>
      <ul class="cards">
{cards}
      </ul>
    </section>""")
    appendices = "\n".join(
        f"""      <li><strong>Appendix {esc(a['letter'])}. {rich(a['title'])}.</strong> {rich(a['summary'])}</li>"""
        for a in cfg.get("appendices", []))
    description = "\n".join(f"    <p>{rich(p)}</p>" for p in cfg["description"])
    size = f" <span class=\"btn-note\">{human_size(book.stat().st_size)}</span>" if book else ""
    corrections_mail = mailto(cfg["email"], f"Correction: {plain(cfg['title'])}",
                              "Chapter and section (or page):\n\nWhat it says now:\n\n"
                              "What it should say:\n")
    any_matlab = any((ROOT / c["matlab"]).is_file() for c in chapters if c.get("matlab"))
    matlab_line = ("\n        <li><strong>MATLAB.</strong> Chapters with a MATLAB version of the "
                   "script list it on their page. Run it from the folder it is saved in.</li>"
                   if any_matlab else "")
    n_ch, n_app = len(chapters), len(cfg.get("appendices", []))
    if cfg.get("book_public", True):
        book_button = (f'<a class="btn btn-primary" href="{esc(cfg["book_pdf"])}">'
                       f'Download the notes (PDF){size}</a>')
    else:
        book_button = ""
    book_note = ("" if cfg.get("book_public", True) else
                 " The notes themselves are shared with students of the course through the course page.")
    body = f"""<section class="hero">
  <div class="wrap hero-inner">
    <div class="hero-text">
      <h1>{rich(cfg['title'])}</h1>
      <p class="subtitle">{rich(cfg['subtitle'])}</p>
      <p class="byline"><span class="byline-name">{esc(cfg['author'])}</span><br>
        {esc(cfg['position'])}<br>{esc(cfg['affiliation'])}</p>
      <p class="lead">{rich(cfg['lead'])}</p>
      <p class="actions">
        {book_button}
        <a class="btn btn-secondary" href="#chapters">Browse the chapters</a>
      </p>
      <p class="hero-meta">{n_ch} chapters and {n_app} appendices, each chapter with a notebook you can run.{book_note}</p>
    </div>
    <div class="hero-art" aria-hidden="true">{logo_svg(176)}</div>
  </div>
</section>

<div class="wrap">
  <section class="prose about" aria-labelledby="about">
    <h2 id="about">About these notes</h2>
{description}
    <p><strong>What the reader needs.</strong> {rich(cfg['prerequisites'])}</p>
    <p class="credit">{rich(cfg['credit'])}</p>
  </section>

  <section id="chapters" class="chapters" aria-labelledby="chapters-h">
    <h2 id="chapters-h">Chapters</h2>
    <p class="section-intro">Each chapter page has the notebook (in Colab or right in your browser),
      the Python script, and a link for reporting a correction.</p>
{chr(10).join(groups)}
    <section class="part" aria-labelledby="part-app">
      <h3 id="part-app"><span class="part-span">Appendices</span> In the PDF</h3>
      <ul class="appendix-list">
{appendices}
      </ul>
    </section>
  </section>

  <section id="notebooks" class="prose howto" aria-labelledby="notebooks-h">
    <h2 id="notebooks-h">How to use the notebooks</h2>
    <p>Every chapter has a notebook that reproduces its main examples. It uses only
      <code>numpy</code>, <code>scipy</code> and <code>matplotlib</code>. There are three ways to run it.
      Each one starts from the chapter's page.</p>
    <ol class="steps">
      <li><strong>Open in Colab.</strong> The notebook runs on Google's servers. You need a
        Google account. To keep your changes, use <em>File, Save a copy in Drive</em>.</li>
      <li><strong>Run in your browser (no login).</strong> The notebook runs inside your web
        browser, on your own device, with nothing to install. The first start loads Python
        and takes a minute or so. Changes stay in that browser; use <em>File, Download</em> to
        keep a copy.</li>
      <li><strong>On your own computer.</strong> Download the notebook or the Python script.
        You need Python 3.9 or later and three packages:
        <pre><code>pip install numpy scipy matplotlib</code></pre>
        Then open the notebook in Jupyter, or run the script, for example
        <code>python3 ch01_laser_loop.py</code>. The scripts print the numbers of the chapter
        and save their figures in a <code>figures</code> folder.</li>{matlab_line}
    </ol>
    <p><strong>Then change something.</strong> Run all the cells from the top once (in Colab,
      <em>Runtime, Run all</em>). Then change a gain or the sampling period <i>T</i>, run the
      cells again, and compare with the chapter.</p>
  </section>

  <section id="corrections" class="callout" aria-labelledby="corrections-h">
    <h2 id="corrections-h">Corrections</h2>
    <p>I will be grateful to readers who report errors, unclear passages or suggestions for
      improvement. Please write to
      <a href="mailto:{esc(cfg['email'])}">{esc(cfg['email'])}</a>, with the chapter and the
      section (or page).</p>
    <p><a class="btn btn-secondary" href="{esc(corrections_mail)}">Report a correction</a></p>
  </section>
</div>"""
    return page(cfg, title=cfg["title"], description=cfg["lead"], prefix=prefix, body=body,
                home_href="./")


# --------------------------------------------------------------------------- chapter pages
def chapter_page(cfg, ch, prev_ch, next_ch, warnings):
    prefix = "../"
    slug, n = ch["slug"], ch["number"]
    user, repo, branch = cfg["github_user"], cfg["repo"], cfg.get("branch", "main")
    colab = (f"https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/"
             f"{ch['notebook']}")
    lite = f"{prefix}lite/lab/index.html?path={Path(ch['notebook']).name}"

    def exists(key):
        rel = ch.get(key)
        if rel and (ROOT / rel).is_file():
            return rel
        if rel and key != "matlab":
            warnings.append(f"{slug}: {key} file not found: {rel}")
        return None

    nb, py, ml = exists("notebook"), exists("python"), exists("matlab")
    downloads = []
    if nb:
        downloads.append(f'<a class="btn btn-quiet" href="{prefix}{esc(nb)}" download>'
                         f'Download notebook <span class="btn-note">(.ipynb)</span></a>')
    if py:
        downloads.append(f'<a class="btn btn-quiet" href="{prefix}{esc(py)}" download>'
                         f'Python script <span class="btn-note">(.py)</span></a>')
    if ml:
        downloads.append(f'<a class="btn btn-quiet" href="{prefix}{esc(ml)}" download>'
                         f'MATLAB script <span class="btn-note">(.m)</span></a>')
    if ch.get("video"):
        downloads.append(f'<a class="btn btn-quiet" href="{esc(ch["video"])}">Video</a>')
    downloads_html = "\n        ".join(downloads)

    run = ""
    if nb:
        run = f"""
    <section class="panel" aria-labelledby="run-h">
      <h2 id="run-h">Run the examples</h2>
      <p class="actions">
        <a class="btn btn-primary" href="{esc(colab)}">Open in Colab</a>
        <a class="btn btn-secondary" href="{esc(lite)}">Run in your browser (no login)</a>
      </p>
      <p class="note">Colab needs a Google account. The browser version needs nothing; its first
        start takes a minute or so. <a href="{prefix}#notebooks">How to use the notebooks</a></p>
      <p><strong>Hands-on.</strong> The last section of the notebook goes with the Hands-on
        exercises at the end of the chapter. Write your prediction on paper first, then use the
        sliders to check it.</p>
      <p class="actions actions-small">
        {downloads_html}
      </p>
    </section>"""
    elif downloads:
        run = f"""
    <section class="panel" aria-labelledby="run-h">
      <h2 id="run-h">Files</h2>
      <p class="actions actions-small">
        {downloads_html}
      </p>
    </section>"""

    reproduces = ""
    if ch.get("reproduces"):
        reproduces = f"""
    <section class="prose" aria-labelledby="code-h">
      <h2 id="code-h">What the code reproduces</h2>
      <p>{rich(ch['reproduces'])}</p>
    </section>"""

    figure = ""
    fig = ch.get("figure")
    if fig and (ROOT / fig).is_file():
        figure = f"""
    <figure class="figure">
      <img src="{prefix}{esc(fig)}" alt="{esc(plain(ch.get('figure_alt', '')))}" loading="lazy">
      <figcaption>A figure drawn by this chapter's notebook.</figcaption>
    </figure>"""
    elif fig:
        warnings.append(f"{slug}: figure not found: {fig}")

    subject = f"Correction: Chapter {n}"
    mail = mailto(cfg["email"], subject,
                  f"Chapter {n}, section (or page):\n\nWhat it says now:\n\nWhat it should say:\n")
    permalink = f"w3id.org/{cfg['w3id_prefix']}/{slug}"

    pager = []
    if prev_ch:
        pager.append(f'<a class="pager-prev" href="{prefix}{prev_ch["slug"]}/" rel="prev">'
                     f'<span class="pager-label">Previous: Chapter {prev_ch["number"]}</span>'
                     f'<span class="pager-title">{rich(prev_ch["title"])}</span></a>')
    else:
        pager.append(f'<a class="pager-prev" href="{prefix}"><span class="pager-label">Back</span>'
                     f'<span class="pager-title">All chapters</span></a>')
    if next_ch:
        pager.append(f'<a class="pager-next" href="{prefix}{next_ch["slug"]}/" rel="next">'
                     f'<span class="pager-label">Next: Chapter {next_ch["number"]}</span>'
                     f'<span class="pager-title">{rich(next_ch["title"])}</span></a>')
    else:
        pager.append(f'<a class="pager-next" href="{prefix}#chapters"><span class="pager-label">'
                     f'Back</span><span class="pager-title">All chapters</span></a>')

    body = f"""<div class="wrap chapter">
  <nav class="crumbs" aria-label="Breadcrumb">
    <a href="{prefix}">Home</a> <span aria-hidden="true">/</span>
    <a href="{prefix}#chapters">Chapters</a> <span aria-hidden="true">/</span>
    <span aria-current="page">Chapter {n}</span>
  </nav>
  <article>
    <header class="chapter-head">
      <p class="eyebrow">Chapter {n}</p>
      <h1>{rich(ch['title'])}</h1>
      <p class="lead">{rich(ch['summary'])}</p>
    </header>{run}{reproduces}{figure}
    <section class="callout" aria-labelledby="fix-h">
      <h2 id="fix-h">Found an error?</h2>
      <p>Please report errors and unclear passages in this chapter. Give the section (or page)
        and what it should say.</p>
      <p><a class="btn btn-secondary" href="{esc(mail)}">Report a correction</a></p>
    </section>
    <p class="permalink">Permanent address of this page: <span class="url">{esc(permalink)}</span></p>
  </article>
  <nav class="pager" aria-label="Chapter navigation">
    {(chr(10) + '    ').join(pager)}
  </nav>
</div>"""
    return page(cfg, title=f"Chapter {n}. {ch['title']}", description=ch["summary"],
                prefix=prefix, body=body, home_href=prefix, canonical_path=f"{slug}/")


def not_found_page(cfg):
    base = cfg["base_path"]
    body = f"""<div class="wrap notfound prose">
  <h1>Page not found</h1>
  <p>There is no page at this address. It may have moved.</p>
  <p class="actions">
    <a class="btn btn-primary" href="{esc(base)}">Go to the home page</a>
    <a class="btn btn-secondary" href="{esc(base)}#chapters">See all chapters</a>
  </p>
  <p>Chapter pages have addresses such as <span class="url">…/ch08/</span>.</p>
</div>"""
    return page(cfg, title="Page not found", description=cfg["lead"], prefix=base, body=body,
                home_href=base)


# --------------------------------------------------------------------------- build
def find_book(cfg, override):
    for candidate in ([Path(override)] if override else []) + [
            ROOT / cfg["book_pdf"], (ROOT / cfg.get("book_pdf_local", "")).resolve()]:
        if candidate and candidate.is_file() and candidate.suffix == ".pdf":
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "site.yml"))
    ap.add_argument("--out", default=str(HERE / "_site"))
    ap.add_argument("--book", help="PDF to publish as the notes (default: book_pdf in site.yml)")
    ap.add_argument("--no-copy", action="store_true",
                    help="write the pages only; do not copy notebooks, scripts and the PDF")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings = []
    if cfg["github_user"] == "USER":
        warnings.append("github_user is still USER in site.yml: the Colab links will not work")

    book = find_book(cfg, args.book) if cfg.get("book_public", True) else None
    if not book and cfg.get("book_public", True):
        warnings.append(f"book PDF not found ({cfg['book_pdf']}): the download button will not work")

    (out / "style.css").write_text((HERE / "style.css").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "favicon.svg").write_text(logo_svg(64, decorative=False) + "\n", encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    (out / "index.html").write_text(home_page(cfg, book), encoding="utf-8")
    (out / "404.html").write_text(not_found_page(cfg), encoding="utf-8")

    chapters = sorted(cfg["chapters"], key=lambda c: c["number"])
    for i, ch in enumerate(chapters):
        prev_ch = chapters[i - 1] if i > 0 else None
        next_ch = chapters[i + 1] if i + 1 < len(chapters) else None
        d = out / ch["slug"]
        d.mkdir(exist_ok=True)
        (d / "index.html").write_text(chapter_page(cfg, ch, prev_ch, next_ch, warnings),
                                      encoding="utf-8")

    if not args.no_copy:
        for folder in ("notebooks", "python", "matlab"):
            src = ROOT / folder
            if src.is_dir():
                shutil.copytree(src, out / folder, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", ".*"))
        if book:
            dest = out / cfg["book_pdf"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(book, dest)

    print(f"Built {len(chapters)} chapter pages, index.html and 404.html in {out}")
    for w in warnings:
        print(f"  note: {w}")


if __name__ == "__main__":
    sys.exit(main())
