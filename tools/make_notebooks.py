"""Build one Jupyter notebook per chapter from the companion scripts in ../python.

Each script's docstring becomes the first (text) cell; the code is split into cells at
top-level blocks. The notebooks run unchanged in Google Colab or a local Jupyter.
Run:  python3 make_notebooks.py
"""
import ast, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "python"
OUT = HERE.parent / "notebooks"
OUT.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def split_cells(body):
    tree = ast.parse(body)
    lines = body.splitlines(True)
    starts = []
    for node in tree.body:
        start = node.lineno - 1
        if hasattr(node, "decorator_list") and node.decorator_list:
            start = node.decorator_list[0].lineno - 1
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        starts.append(start)
    cells, current = [], []
    groups = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(lines)
        groups.append("".join(lines[s:e]))
    # merge consecutive small statements (imports, constants) into one cell
    buf = ""
    for g in groups:
        big = g.lstrip().startswith(("def ", "class ", "#", "@", "if __name__", "for ", "with "))
        if big and buf.strip():
            cells.append(buf); buf = ""
        buf += g
        if big:
            cells.append(buf); buf = ""
    if buf.strip():
        cells.append(buf)
    return [c for c in cells if c.strip()]


def explore_cells(py):
    """The chapter's interactive section (python/explore/chNN_explore.py), if there is one."""
    ex = SRC / "explore" / f"{py.stem.split('_')[0]}_explore.py"
    if not ex.exists():
        return []
    text = ex.read_text()
    m = re.match(r'\s*"""(.*?)"""\n', text, re.S)
    doc, body = (m.group(1).strip(), text[m.end():]) if m else ("", text)
    setup = ("# In the browser (JupyterLite) the sliders need one extra package.\n"
             "import sys\n"
             "if \"pyodide\" in sys.modules:\n"
             "    import piplite\n"
             "    await piplite.install(\"ipywidgets\")\n")
    return [md("## Explore\n\n" + doc + "\n"), code(setup), code(body)]


def handson_cells(py):
    """The Hands-on section (python/handson/chNN_handson.py), if the chapter has one."""
    ex = SRC / "handson" / f"{py.stem.split('_')[0]}_handson.py"
    if not ex.exists():
        return []
    text = ex.read_text()
    m = re.match(r'\s*r?"""(.*?)"""\n', text, re.S)
    doc, body = (m.group(1).strip(), text[m.end():]) if m else ("", text)
    setup = ("# In the browser (JupyterLite) the sliders need one extra package.\n"
             "import sys\n"
             "if \"pyodide\" in sys.modules:\n"
             "    import piplite\n"
             "    await piplite.install(\"ipywidgets\")\n")
    return [md("## Hands-on\n\n" + doc + "\n"), code(setup)] + [code(c) for c in split_cells(body)]


def convert(py):
    text = py.read_text()
    m = re.match(r'\s*"""(.*?)"""\n', text, re.S)
    doc, body = (m.group(1), text[m.end():]) if m else ("", text)
    body = re.sub(r'^matplotlib\.use\("Agg"\).*\n', "", body, flags=re.M)
    body = body.replace("Path(__file__).resolve().parent", "Path.cwd()")
    body = body.replace("os.path.dirname(os.path.abspath(__file__))", "os.getcwd()")
    title = doc.strip().splitlines()[0] if doc.strip() else py.stem
    rest = "\n".join(doc.strip().splitlines()[1:]).strip()
    rest = re.sub(r"\n\s*Run:.*", "", rest, flags=re.S)
    cells = [md(f"# {title}\n\n*Computers in the Loop: Digital Control from Sampling to Networked "
                f"Agents*: EECE 5610 lecture notes prepared and revised by Milad Siami.\n\n"
                "Chapters 1–11 of the notes are based on the lecture notes of Prof. Mario Sznaier; "
                "all credits are in `ATTRIBUTION.md`.\n\n"
                f"{rest}\n\nRun the cells from top to bottom "
                "(in Colab: *Runtime > Run all*). Figures are also saved to a `figures` folder.\n")]
    cells += [code(c) for c in split_cells(body)]
    cells += explore_cells(py)
    cells += handson_cells(py)
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
          "language": "python"}, "language_info": {"name": "python"},
          "authors": [{"name": "Milad Siami"}]}, "nbformat": 4, "nbformat_minor": 5}
    name = py.stem.split("_")[0]          # ch01, ch02, ...
    out = OUT / f"{name}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def activity_notebooks():
    """Self-contained notebooks for the eleven added classroom investigations."""
    text = (SRC / 'activities' / 'experiments.py').read_text()
    body = text[text.index('from pathlib import Path'):text.index("if __name__=='__main__':")]
    body = body.replace("OUT=Path(os.environ.get('EECE5610_ACTIVITY_OUTPUT',str(Path(__file__).resolve().parent/'figures')))",
                        "OUT=Path.cwd()/'activity_figures'")
    body = body.replace("matplotlib.use('Agg')", "# Use the notebook plotting backend.")
    destination = OUT / 'activities'
    destination.mkdir(exist_ok=True)
    for n in range(1,12):
        related = (' The nominal model is related to textbook Example 6.1.' if n == 1 else
                   ' The frequency parameters also occur in textbook Problem 3.7-7; the investigation here asks a different question.' if n == 6 else '')
        cells = [md(f'# Chapter {n}: computational activity\n\n'
                    'EECE 5610 lecture notes, prepared and revised by Milad Siami.\n\n'
                    'Work the activity at the end of the chapter on paper first. '
                    'State your prediction and assumptions before running the experiment.'+related+'\n\n'
                    'Run the setup cells, inspect the named case function, and change one parameter at a time. '
                    'All experiments are mathematical simulations; no hardware is connected.'),
                 code('%matplotlib inline')]
        cells += [code(c) for c in split_cells(body)]
        cells += [md(f'## Run case {n}\n\nChange the keyword arguments below to investigate the assigned variation. '
                     f'Use `help(case{n:02d})` or inspect its definition for available parameters.'),
                  code(f'result = run_case({n})\nfrom IPython.display import display, Image\ndisplay(Image(filename=str(OUT / "case{n:02d}.png")))\nresult')]
        nb = {'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
               'language_info':{'name':'python'},'authors':[{'name':'Milad Siami'}]},'nbformat':4,'nbformat_minor':5}
        out=destination/f'activity_ch{n:02d}.ipynb';out.write_text(json.dumps(nb,indent=1));print(out.name)


if __name__ == "__main__":
    for py in sorted(SRC.glob("ch[0-9][0-9]_*.py")):
        print(convert(py).name)
    activity_notebooks()
