# Companion website

A small static website for the lecture notes *Computers in the Loop: Digital Control from Sampling to
Networked Agents*: a home page with the PDF and the chapter list, and one page per chapter with
its notebook (in Colab, or in the browser with no login), the Python and MATLAB scripts, and a
link for reporting a correction. Each chapter page lives at `.../chNN/`, which is where the QR
code printed in the notes points.

> **Status:** published at https://siamilab.github.io/computers-in-the-loop/ by the deploy
> workflow (`.github/workflows/deploy.yml`) on every push to `main`. The QR codes in the notes
> reach the chapter pages through https://w3id.org/citl/chNN.

## Files

| File | What it is |
|---|---|
| `site.yml` | Everything that changes: title, author, e-mail, GitHub user and repository, and the chapter list (number, `chNN`, title, one-line summary, notebook, Python and MATLAB files, video link, figure). |
| `build_site.py` | Writes the site to `_site/` from `site.yml`. Standard library only (uses PyYAML if installed). |
| `style.css` | The look: system fonts for the interface, Charter/Georgia for reading, orange accent, phone first, dark mode. |
| `check_links.py` | Checks every relative link, image and `#anchor` in `_site/`, and that the HTML tags are balanced. |
| `../.github/workflows/deploy.yml` | Builds and publishes the site with GitHub Pages on every push to `main`. |

The pages use no JavaScript and load nothing from other servers.

## 1. Set the GitHub user

In `site.yml`, replace `USER` with your GitHub user name:

```yaml
github_user: "your-user-name"
repo: "computers-in-the-loop"
base_path: "/computers-in-the-loop/"   # /<repo>/ on GitHub Pages
```

The Colab buttons use these two names
(`https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/chNN.ipynb`), so
they work only after the repository is public. If you rename the repository, change `repo` and
`base_path` together. While `github_user` is still `USER`, the build prints a reminder.

Other things to edit in `site.yml`: a chapter's `video:` (leave it `""` and the Video button is
not shown), the summaries, and the text of the home page.

## 2. Preview on your computer

From this folder:

```
python3 build_site.py && python3 -m http.server -d _site
```

then open <http://localhost:8000>. The build copies the notebooks, scripts, figures and the notes
PDF into `_site/` so the download buttons work. The PDF is taken from
`../book/computers-in-the-loop.pdf`, or, if that is missing, from the compiled notes
`../../eece5610-course-notes.pdf`. Another PDF can be given with `--book file.pdf`.

Two things work only on the published site: the Colab buttons (they read the notebook from
GitHub) and "Run in your browser", which needs the `lite/` folder that the workflow builds with
JupyterLite. To try that part locally as well:

```
pip install jupyterlite-core jupyterlite-pyodide-kernel
jupyter lite build --contents ../notebooks --output-dir _site/lite
```

Check the links after any change:

```
python3 check_links.py
```

It should end with `OK`. Links into `lite/` are counted but not treated as errors.

A chapter's MATLAB button appears only when its file (for example `matlab/ch01_laser_loop.m`)
exists, and its figure only when the PNG exists, so rebuild after adding scripts or rerunning
them.

## 3. Deploy with GitHub Pages

The repository is the `companion/` folder: `notebooks/`, `python/`, `matlab/`, `website/`,
`.github/`, the READMEs, and the PDF of the notes.

1. Create a public repository `computers-in-the-loop` on GitHub.
2. Put the compiled notes at `book/computers-in-the-loop.pdf` in the repository (copy
   `eece5610-course-notes.pdf` and rename it). The workflow stops with an error if it is missing.
3. Commit the `companion/` folder and push it to `main`. `website/_site/` is ignored by git.
4. On GitHub: Settings, Pages, Build and deployment, Source: **GitHub Actions**.
5. The workflow then runs on every push to `main` (or from the Actions tab, "Run workflow"). It
   - installs `jupyterlite-core` and `jupyterlite-pyodide-kernel`,
   - runs `jupyter lite build --contents notebooks --output-dir _site/lite`,
   - runs `build_site.py`,
   - copies `notebooks/`, `python/`, `matlab/` and the PDF of the notes into `_site/`,
   - checks the links, and publishes `_site/` with `actions/upload-pages-artifact` and
     `actions/deploy-pages`.
6. The site appears at `https://<user>.github.io/computers-in-the-loop/`.

The first run of the workflow has not happened yet, so check its log on that first push.

## 4. The permanent address in the notes (w3id)

The notes print `https://w3id.org/citl/chNN` (`w3id_prefix: "citl"`) once `\codeurl` is set in
`2026-fall/lectures/board-notes-preamble.tex`. To make that address forward to the site, add a
folder `citl/` to the w3id repository (<https://github.com/perma-id/w3id.org>) by pull request,
with this `.htaccess`:

```
RewriteEngine on
RewriteRule ^(.*)$ https://<user>.github.io/computers-in-the-loop/$1 [R=302,L]
```

Then set `\providecommand{\codeurl}{https://w3id.org/citl}` in the preamble and rebuild the notes.
If the site ever moves, only the `.htaccess` changes; the printed QR codes keep working.
