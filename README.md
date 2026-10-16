# 🐾 Deep Learning in the Kalahari

A one-hour, hands-on session on chapters 1–8 of Simon J.D. Prince's *Understanding Deep Learning* (MIT Press; free at [udlbook.github.io](https://udlbook.github.io/udlbook/)), written for the Transition Bio ML reading group.

The notebook covers the book's maths and rebuilds each idea from scratch in NumPy, with auto-checked puzzles set in a meerkat field station. All data are synthetic and generated inside the notebook, so every answer can be checked.

## The session (60 min)

| Time | What | Material |
|---|---|---|
| 0–12 min | Slides: one idea per chapter (10 slides) | [deck](slides/TBio_UDL_Ch1-8_OneHour.pptx) |
| 12–55 min | Hands-on: the core puzzle path | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/transitionbio/udl-ch1-8-tutorial/blob/main/notebooks/TBio_UDL_Ch1-8_OneHour.ipynb) |
| 55–60 min | Wrap-up and questions | |

**Core path (≈ 40 min):** 10 puzzles and 2 optional challenges.

| Part | Chapter | Puzzles | Time |
|---|---|---|---|
| 2 | Supervised learning | 2.1 gradient of the least squares loss | 3 min |
| 3 | Shallow networks | 3.1 joints & slopes · 🏆 dig the burrow | 8 min |
| 4 | Deep networks | 4.1 predict the folds | 3 min |
| 5 | Loss functions | 5.3 BCE gradient · 5.4 stable softmax · 5.6 which way is the burrow? | 9 min |
| 6 | Fitting models | 6.4 momentum / Nesterov / Adam · 🏆 race to the burrow | 8 min |
| 7 | Gradients & init | 7.1 backprop by hand | 6 min |
| 8 | Measuring performance | 8.1 · 8.4 quick questions | 2 min |

Every chapter's recap and demo plots are still in the notebook. Puzzles outside the core path appear as collapsed ⚙️ *ready-made helper* cells, so everything runs.

### Running it
- Open the notebook in Colab, then **Runtime → Run all**. Unfilled puzzles report "still a TODO"; everything else just works. Then go top to bottom.
- 🔑 hidden solution cells are there if someone gets stuck (run them so later cells still work).
- `progress()` at the end shows each person what they've solved. It's private to them: there are no scores or rankings, and nothing is recorded or shared.
- CPU-only Colab is fine; the whole notebook executes in under a minute.

> **Private repo / Colab access:** Colab can only open this notebook once Colab is allowed to read `transitionbio` repos: authorise GitHub in Colab, then grant *Colaboratory* access to the `transitionbio` org (github.com/settings/applications). Alternatively, download the `.ipynb` and use **File → Upload notebook** in Colab.

## Extended version

[`extended/`](extended/) has the full three-session course for anyone who wants more. It has 35 puzzles, every derivation, a protein-condensate capstone, and a 7–9 slide deck per session.

| # | Topic | Chapters | Notebook | Slides |
|---|---|---|---|---|
| 1 | Models | 1–4 | [notebook](extended/notebooks/TBio_UDL_S1_Models_Ch1-4.ipynb) | [deck](extended/slides/TBio_UDL_S1_Models_Ch1-4.pptx) |
| 2 | Losses and training | 5–6 | [notebook](extended/notebooks/TBio_UDL_S2_Losses_Training_Ch5-6.ipynb) | [deck](extended/slides/TBio_UDL_S2_Losses_Training_Ch5-6.pptx) |
| 3 | Backprop and performance (+ capstone) | 7–8 | [notebook](extended/notebooks/TBio_UDL_S3_Backprop_Performance_Ch7-8.ipynb) | [deck](extended/slides/TBio_UDL_S3_Backprop_Performance_Ch7-8.pptx) |

## Rebuilding / editing

All notebooks are generated from `src/`. Each `p*.py` file is one part of the book. `p0_intro.py` holds the intro and setup, and `build.py` has the core-path puzzle list (`CORE`).

```bash
cd src
python build.py ../notebooks student hour             # one-hour notebook
python build.py ../extended/notebooks student series  # three-session notebooks
python build.py /tmp/check test hour                  # verification build (solutions before checks)
python run.py /tmp/check/TBio_UDL_Ch1-8_OneHour.ipynb /tmp/check/out.ipynb   # every ✅ should print
```

Requires `nbformat` and `nbclient` to build and run, plus `numpy`, `matplotlib` and `torch` to execute.

## Credits

Content follows Prince, *Understanding Deep Learning* (MIT Press, 2023; CC-BY-NC-ND), including its notation, equation and figure numbers, and selected end-of-chapter problems. The meerkat call dataset is modelled on MNIST-1D (Greydanus, 2020). All datasets are synthetic; the condensate capstone is a toy and is not Transition Bio data.

Internal training material: Confidential and Proprietary, Transition Bio.
