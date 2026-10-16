# 🐾 Deep Learning in the Kalahari

A three-session tutorial on chapters 1–8 of Simon J.D. Prince's *Understanding Deep Learning* (MIT Press; free at [udlbook.github.io](https://udlbook.github.io/udlbook/)), written for the Transition Bio ML reading group.

Each session is a short deck (7–9 slides, about 20 minutes) followed by **60–70 minutes of hands-on puzzles** in Colab. The notebooks derive the book's maths step by step and rebuild each idea from scratch in NumPy before switching to PyTorch. Each step comes with auto-checked puzzles set in a meerkat field station. All data are synthetic and generated inside the notebooks, so every answer can be checked.

## Sessions

| # | Topic | Chapters | Notebook | Slides |
|---|---|---|---|---|
| 1 | Models | 1–4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/transitionbio/udl-ch1-8-tutorial/blob/main/notebooks/TBio_UDL_S1_Models_Ch1-4.ipynb) | [deck](slides/TBio_UDL_S1_Models_Ch1-4.pptx) |
| 2 | Losses and training | 5–6 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/transitionbio/udl-ch1-8-tutorial/blob/main/notebooks/TBio_UDL_S2_Losses_Training_Ch5-6.ipynb) | [deck](slides/TBio_UDL_S2_Losses_Training_Ch5-6.pptx) |
| 3 | Backprop and performance (+ capstone) | 7–8 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/transitionbio/udl-ch1-8-tutorial/blob/main/notebooks/TBio_UDL_S3_Backprop_Performance_Ch7-8.ipynb) | [deck](slides/TBio_UDL_S3_Backprop_Performance_Ch7-8.pptx) |

> **Private repo:** the Colab badges work after you authorise GitHub in Colab (File → Open notebook → GitHub → tick *Include private repos*). Or download the `.ipynb` and use File → Upload notebook.

### What's in each session
1. **Models:** supervised / unsupervised / RL; 1D linear regression (gradients, closed form, generative vs discriminative); shallow ReLU networks (joints, activation patterns, universal approximation, Zaslavsky's region count); deep networks (folding, matrix form, depth efficiency).
2. **Losses and training:** the maximum-likelihood recipe; least squares, heteroscedastic regression, binary and multiclass cross-entropy, Poisson and von Mises losses; GD on the non-convex Gabor model, SGD, momentum, Nesterov and Adam implemented from the equations.
3. **Backprop and performance:** backprop by hand and for any MLP (checked against autograd); He initialisation and the Fig 7.7 recreation; the noise/bias/variance decomposition measured by simulation; double descent; the curse of dimensionality. It finishes with a capstone: a toy protein-condensate phase-diagram classifier.

## How the notebooks work

| | |
|---|---|
| 🧩 ⭐ / ⭐⭐ / ⭐⭐⭐ | Puzzles: quick / medium / stretch. Fill in the `TODO`, then run the 🧪 check cell |
| 🔑 | Hidden solution cell (double-click). Have a go first; run it if stuck so later cells still work |
| 🏆 | Open challenges with bronze / silver / gold targets to aim for |
| 📦 | Sessions 2–3: a carry-over cell re-creates what's needed from earlier sessions |

`progress()` at the end shows each person which puzzles they've solved and which are left. It's private to them: there are no scores or rankings, and nothing is recorded or shared.

**Facilitator tips:** each notebook's intro lists suggested minutes per part and the *stretch* puzzles to leave for homework if time runs short. CPU-only Colab is fine; each notebook executes end to end in a few minutes.

## Rebuilding / editing

The notebooks are generated from `src/`. Each `p*.py` file is one part of the book, `p0_intro.py` holds the setup cell and score keeper, and `p_carry.py` holds the carry-over cells.

```bash
cd src
python build.py ../notebooks          # student notebooks (solutions hidden after each check)
python build.py /tmp/test test        # verification build: solutions placed before checks
for f in /tmp/test/*.ipynb; do python run.py "$f" "${f%.ipynb}_out.ipynb"; done   # every ✅ should print
```

Requires `nbformat` and `nbclient` to build and run, plus `numpy`, `matplotlib` and `torch` to execute.

## Credits

Content follows Prince, *Understanding Deep Learning* (MIT Press, 2023; CC-BY-NC-ND), including its notation, equation and figure numbers, and selected end-of-chapter problems. The meerkat call dataset is modelled on MNIST-1D (Greydanus, 2020). All datasets are synthetic; the condensate capstone is a toy and is not Transition Bio data.

Internal training material: Confidential and Proprietary, Transition Bio.
