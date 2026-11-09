# Computers in the Loop: companion notebooks and code

Interactive examples for the lecture notes **Computers in the Loop: Digital Control from Sampling
to Networked Agents** by Milad Siami (Northeastern University), written for the graduate course
EECE 5610 Digital Control Systems.

**Website:** https://siamilab.github.io/computers-in-the-loop/
**Permanent chapter links (the QR codes in the notes):** `https://w3id.org/citl/ch01` ... `ch16`

The notes themselves are shared with students of the course through the course page. This
repository holds the material that runs: one notebook per chapter, the Python and MATLAB scripts
behind it, and the website.

## Run a chapter

| Way | What you need |
|---|---|
| **Google Colab** | A Google account. Open `notebooks/chNN.ipynb` from the chapter page and choose *Runtime > Run all*. |
| **In your browser** | Nothing. The chapter page has a "Run in your browser" button (JupyterLite); the first start takes a minute. |
| **On your computer** | Python 3.9 or later. For the scripts: `python -m pip install -r requirements.txt`, then `python run_chapter.py 6`. For the notebooks (adds Jupyter and the sliders): `python -m pip install -r requirements-notebooks.txt`, then `jupyter notebook notebooks/ch06.ipynb`. |
| **MATLAB** | `matlab/chNN_*.m`, run from the `matlab` folder. |

Each notebook has three parts:

1. **The chapter's examples**, computed again from the model, with the figures of the notes.
2. **Explore**: sliders for the sampling period, the gains or the model, to see what changes.
3. **Hands-on**: one section for each of the four exercises at the end of the chapter
   (*Predict*, *Spot the error*, *Explore*, *Design*). Work the exercise on paper first and write
   down your prediction; then use the notebook to check it.

Chapters 1–11 also have an activity notebook in `notebooks/activities/`, which goes with the
computational activity at the end of the chapter.

## Chapters

| | Chapter | Notebook |
|---|---|---|
| 1 | Introduction to Digital Control | `notebooks/ch01.ipynb` |
| 2 | Block Diagrams and Mason's Gain Formula | `notebooks/ch02.ipynb` |
| 3 | Discrete-Time Systems and the z-Transform | `notebooks/ch03.ipynb` |
| 4 | Difference Equations and the Inverse z-Transform | `notebooks/ch04.ipynb` |
| 5 | Realizations, State Variables, and the Sample-and-Hold | `notebooks/ch05.ipynb` |
| 6 | The Starred Transform, Aliasing, and the Zero-Order Hold | `notebooks/ch06.ipynb` |
| 7 | Data Reconstruction and Data Holds | `notebooks/ch07.ipynb` |
| 8 | Open-Loop Sampled-Data Systems | `notebooks/ch08.ipynb` |
| 9 | The Modified z-Transform and Closed-Loop Systems | `notebooks/ch09.ipynb` |
| 10 | Closed-Loop Time Response and Steady-State Accuracy | `notebooks/ch10.ipynb` |
| 11 | Stability of Sampled-Data Systems | `notebooks/ch11.ipynb` |
| 12 | Root Locus and Frequency Response of Sampled-Data Loops | `notebooks/ch12.ipynb` |
| 13 | Designing Digital Controllers | `notebooks/ch13.ipynb` |
| 14 | State-Space Design: Pole Placement, Observers, and Optimal Control | `notebooks/ch14.ipynb` |
| 15 | Implementing Digital Controllers | `notebooks/ch15.ipynb` |
| 16 | How Fast Must Networked Agents Sample? | `notebooks/ch16.ipynb` |

## Repository layout

```
notebooks/          one notebook per chapter (built from python/), activities/
python/             chapter scripts; explore/ and handson/ hold the slider sections
matlab/             MATLAB versions of the chapter scripts
run_chapter.py      run one chapter, or all of them
tools/              make_notebooks.py rebuilds notebooks/ from python/
website/            build_site.py and site.yml for the GitHub Pages site
w3id/               the permanent-address redirect behind the QR codes
```

After editing a script in `python/`, run `python tools/make_notebooks.py` to rebuild the notebooks.

## Credits

Chapters 1–11 of the notes are based on the lecture notes of Prof. Mario Sznaier. Several worked
examples in Chapters 6–11 are adapted from the course textbook, C. L. Phillips, H. T. Nagle and
A. Chakrabortty, *Digital Control System Analysis and Design*, 4th ed., Pearson, 2015; the notes
mark each one. Chapter 16 is based on Y.-M. Huang, A. C. B. de Oliveira, D. Murugan and M. Siami,
*IEEE Open Journal of Control Systems*, 2023. See `ATTRIBUTION.md`.

## License

The code (`.py`, `.m`, and the code cells of the notebooks) is released under the MIT License,
see `LICENSE`. The text of the notebooks and of the website is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), see `LICENSE-CONTENT.md`.

## Corrections

Please report errors or suggestions to siami.assistant@gmail.com, with the chapter and the
notebook cell.
