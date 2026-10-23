# 🐾 Deep Learning in the Kalahari

A one-hour, hands-on session on chapters 1–8 of Simon J.D. Prince's *Understanding Deep Learning* (MIT Press; free at [udlbook.github.io](https://udlbook.github.io/udlbook/)), written for the Transition Bio ML reading group.

The notebook covers the book's maths and rebuilds each idea from scratch in NumPy, with auto-checked puzzles set in a meerkat field station. All data are synthetic and generated inside the notebook, so every answer can be checked.

## The session (60 min)

| Time | What | Material |
|---|---|---|
| 0–5 min | Slides: welcome and roadmap | [deck](slides/TBio_UDL_Ch1-8_OneHour.pptx) |
| 5–55 min | Chapter by chapter: concept slide → problem slide → solve it in the notebook → solution slide | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/transitionbio/udl-ch1-8-tutorial/blob/main/notebooks/TBio_UDL_Ch1-8_OneHour.ipynb) |
| 55–60 min | Wrap-up and discussion | |

The deck (38 slides) follows the notebook. Each chapter has a concept slide. Each core puzzle has a **problem slide**, which sets out the meerkat scenario in plain words and the task to code, then a **solution slide**: a plain-language explanation for non-ML scientists, followed by the maths and code. Speaker notes carry timings and hints.

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

## Rebuilding / editing

The notebook is generated from `src/`. Each `p*.py` file is one part of the book. `p0_intro.py` holds the intro and setup, and `build.py` has the core-path puzzle list (`CORE`).

```bash
cd src
python build.py ../notebooks student   # the notebook
python build.py /tmp/check test        # verification build (solutions before checks)
python run.py /tmp/check/TBio_UDL_Ch1-8_OneHour.ipynb /tmp/check/out.ipynb   # every ✅ should print
```

Requires `nbformat` and `nbclient` to build and run, plus `numpy`, `matplotlib` and `torch` to execute.

## Credits

Content follows Prince, *Understanding Deep Learning* (MIT Press, 2023; CC-BY-NC-ND), including its notation, equation and figure numbers, and selected end-of-chapter problems. The meerkat call dataset is modelled on MNIST-1D (Greydanus, 2020). All datasets are synthetic and none are Transition Bio data. Emoji illustrations in the deck are [Twemoji](https://github.com/jdecked/twemoji) (CC-BY 4.0).

Internal training material: Confidential and Proprietary, Transition Bio.
