#!/usr/bin/env python3
"""Check the built site: every relative link and image points to a file in _site/, every
#fragment points to an id on its page, and the HTML tags are balanced.

Links into lite/ (the in-browser notebooks) exist only after `jupyter lite build`, so they are
listed separately and do not count as errors.

Usage:  python3 check_links.py [_site] [--base /computers-in-the-loop/]
        (the default base is base_path in site.yml)
Exit code 1 if anything is broken.
"""
import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
        "source", "track", "wbr", "circle", "rect", "path", "polyline"}
DEPLOY_ONLY = ("lite/",)


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.ids, self.stack, self.problems = [], set(), [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if a.get(key) is not None:
                self.links.append((tag, a[key]))
        if tag == "img" and not a.get("alt"):
            self.problems.append("img without alt text")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.problems.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")


def parse(path):
    p = Page()
    p.feed(path.read_text(encoding="utf-8"))
    p.close()
    if p.stack:
        p.problems.append(f"unclosed tags: {p.stack}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?", default=str(Path(__file__).resolve().parent / "_site"))
    ap.add_argument("--base", default=None,
                    help="site path on the server (default: base_path in site.yml); "
                         "links that start with it resolve to the site root")
    args = ap.parse_args()
    root = Path(args.site).resolve()
    if args.base is None:
        from build_site import HERE, load_config
        args.base = load_config(HERE / "site.yml")["base_path"]

    pages = {f: parse(f) for f in sorted(root.rglob("*.html")) if "lite" not in f.relative_to(root).parts}
    errors, deploy_only, checked = [], 0, 0
    for f, p in pages.items():
        name = f.relative_to(root)
        for prob in p.problems:
            errors.append(f"{name}: {prob}")
        for tag, url in p.links:
            parts = urlsplit(url)
            if parts.scheme or url.startswith("//"):
                continue                      # http(s), mailto: checked by eye
            checked += 1
            path = unquote(parts.path)
            if not path:                      # "#id" on the same page
                if parts.fragment and parts.fragment not in p.ids:
                    errors.append(f"{name}: no id '{parts.fragment}' for {url}")
                continue
            if path.startswith(args.base):
                target = root / path[len(args.base):]
            elif path.startswith("/"):
                errors.append(f"{name}: absolute link outside the site path: {url}")
                continue
            else:
                target = (f.parent / path)
            target = target.resolve()
            if target.is_dir() or path.endswith("/"):
                target = target / "index.html"
            rel = target.relative_to(root).as_posix() if root in target.parents or target == root else str(target)
            if rel.startswith(DEPLOY_ONLY):
                deploy_only += 1
                continue
            if not target.is_file():
                errors.append(f"{name}: broken {tag} link {url}")
                continue
            if parts.fragment and target.suffix == ".html":
                tp = pages.get(target) or parse(target)
                if parts.fragment not in tp.ids:
                    errors.append(f"{name}: no id '{parts.fragment}' in {rel} for {url}")

    print(f"Checked {checked} relative links in {len(pages)} pages under {root}")
    if deploy_only:
        print(f"  {deploy_only} of them point into lite/ (built by `jupyter lite build` at deploy time)")
    for e in errors:
        print("  ERROR", e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
