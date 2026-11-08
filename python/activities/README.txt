EECE 5610 — ADDITIONAL CHAPTER 1–11 INVESTIGATIONS
Prepared for the lecture notes by Milad Siami.

From the Companion folder:
  python run_chapter.py 1 --activity
  python run_chapter.py --all --activity

Or from this folder:
  python experiments.py --chapter 1
  python experiments.py --all

The functions case01 through case11 expose the parameters used in each activity.
Plots and numerical records are written to the figures folder beside this file.
Set EECE5610_ACTIVITY_OUTPUT to choose a different output folder if needed.

For notebook use, open notebooks/activities/activity_chNN.ipynb.
Work the activity at the end of the chapter on paper first.
Change one parameter at a time and preserve the model and timing assumptions.
The experiments are mathematical simulations, not hardware-control programs.

Related setups: case01 has nominal dynamics equivalent to the course textbook's
Example 6.1; case06 shares frequencies with textbook Problem 3.7-7 but investigates
a different question. See the source notes in the lecture notes and ATTRIBUTION.md.
