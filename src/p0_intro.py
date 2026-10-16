from nbh import md, code

import nbh
S = nbh.SESSION
INFO = {
 0: dict(name="Chapters 1–8 in an hour", chapters="1–8", maxpts=0, contents=[], stretch="Challenges 3 and 6"),
 1: dict(name="Models", chapters="1–4", maxpts=28,
         contents=[("1", "Introduction", "Ch 1", "supervised, unsupervised and reinforcement learning", 10),
                   ("2", "Supervised learning", "Ch 2", "the Great Pup Weigh-in", 15),
                   ("3", "Shallow neural networks", "Ch 3", "digging burrows with ReLUs", 25),
                   ("4", "Deep neural networks", "Ch 4", "folding the dune sea", 20)],
         stretch="2.3, 3.4, 4.4"),
 2: dict(name="Losses and training", chapters="5–6", maxpts=21,
         contents=[("5", "Loss functions", "Ch 5", "sentinels, alarm calls, scorpions and lost burrows", 35),
                   ("6", "Fitting models", "Ch 6", "race to the burrow (GD, SGD, momentum, Adam)", 35)],
         stretch="5.2, 5.6, 6.3"),
 3: dict(name="Backprop and performance", chapters="7–8", maxpts=23,
         contents=[("7", "Gradients and initialization", "Ch 7", "backprop by hand and whispers down the burrow", 25),
                   ("8", "Measuring performance", "Ch 8", "bias, variance, double descent and the curse of dimensionality", 30),
                   ("9", "Capstone", "Ch 2–8", "from meerkats to molecules (a toy condensate predictor)", 15)],
         stretch="7.4, 8.5 and Challenge 8"),
}[S]
SERIES = "\n".join(
    f"| {'**' if k == S else ''}Session {k}{'**' if k == S else ''} | {nm} | Ch {ch} |"
    for k, nm, ch in [(1, "Models", "1–4"), (2, "Losses and training", "5–6"), (3, "Backprop and performance", "7–8")])
CONTENTS = "\n".join(f"| Part {p} | **{t}** ({c}): {d} | ~{m} min |" for p, t, c, d, m in INFO["contents"])
prev = {0: "", 1: "", 2: "\n> 📦 **Coming from Session 1?** The hidden *carry-over* cell after Setup re-creates everything this session needs from earlier parts, so you don't need to re-run old notebooks.\n",
        3: "\n> 📦 **Coming from Sessions 1–2?** The hidden *carry-over* cell after Setup re-creates everything this session needs from earlier parts (e.g. `deep_forward` from Part 4), so you don't need to re-run old notebooks.\n"}[S]

ONE_HOUR = r"""
# 🐾 Deep Learning in the Kalahari
## *Understanding Deep Learning*, chapters 1–8 in one hour

**Transition Bio · ML Reading Group**

A one-hour, hands-on tour of chapters 1–8 of Simon J.D. Prince's **_Understanding Deep Learning_** (MIT Press; free at [udlbook.github.io](https://udlbook.github.io/udlbook/)). We follow the book's notation, square brackets and all: $\mathbf{y} = \mathbf{f}[\mathbf{x}, \boldsymbol\phi]$.

### ⏱️ The hour
| Time | What |
|---|---|
| 0–12 min | Slides: the big picture, one idea per chapter |
| 12–55 min | **This notebook**: the core puzzle path below |
| 55–60 min | Wrap-up and questions |

### 🦦 The story
You've just joined a field station in the Kalahari that studies a wild meerkat mob. The station has a lot of data and no deep-learning expert, so you're it. All datasets are **synthetic**: they're generated right here, so every answer can be checked automatically.

### 🧭 How it works
| Symbol | Meaning |
|---|---|
| 📖 | Recap of the chapter's maths. Skim it; it's there to come back to |
| 💻 | Demo code: just run it and look at the plot |
| 🧩 | **Core puzzle**: fill in the `TODO`, then run the 🧪 check cell |
| 🏆 | Challenge with bronze / silver / gold targets (do these if you have time) |
| 🔑 | Hidden solution. Have a go first; run it if stuck so later cells still work |
| ⚙️ | Ready-made helper: just run it |

> **Tip:** start with *Runtime → Run all*. Every TODO will report "still a TODO" until you fill it in, and everything else just works. Then go top to bottom through the puzzles.

### 🗺️ The core path (≈ 40 min)
| Part | Chapter | Puzzles | Time |
|---|---|---|---|
| 1 | Introduction | read only | 1 min |
| 2 | Supervised learning | 2.1 gradient of the least squares loss | 3 min |
| 3 | Shallow networks | 3.1 joints & slopes · 🏆 dig the burrow | 8 min |
| 4 | Deep networks | 4.1 predict the folds | 3 min |
| 5 | Loss functions | 5.3 BCE gradient · 5.4 stable softmax · 5.6 which way is the burrow? | 9 min |
| 6 | Fitting models | 6.4 momentum / Nesterov / Adam · 🏆 race to the burrow | 8 min |
| 7 | Gradients & init | 7.1 backprop by hand | 6 min |
| 8 | Measuring performance | 8.1 · 8.4 (quick questions) | 2 min |

⏱ **Short on time?** Skip the 🏆 challenges first. At the end, `progress()` shows what you've solved; it's just for you, and nothing is recorded or shared. The full three-session version of this tutorial, with 35 puzzles and a capstone, is in the repo's `extended/` folder.

*Book figures and problem numbers are cited as e.g. "Fig 3.3" and "P3.8".*
"""

if S == 0:
    md(ONE_HOUR)
else:
  md(f"""
# 🐾 Deep Learning in the Kalahari · Session {S} of 3
## *Understanding Deep Learning*, chapters {INFO['chapters']}: {INFO['name'].lower()}

**Transition Bio · ML Reading Group**

This series works through chapters 1–8 of Simon J.D. Prince's **_Understanding Deep Learning_** (MIT Press; free at [udlbook.github.io](https://udlbook.github.io/udlbook/)) in three sessions. Each one is about **20 min of slides and 60–70 min of hands-on puzzles**. We follow the book's notation, square brackets and all: $\\mathbf{{y}} = \\mathbf{{f}}[\\mathbf{{x}}, \\boldsymbol\\phi]$.

| Session | Topic | Chapters |
|---|---|---|
{SERIES}

---

### 🦦 The story
You've just joined a long-running field station in the Kalahari that studies a wild meerkat mob. The mob is about thirty animals, run by a dominant female (in meerkat society, she's in charge). The station has a lot of data and no deep-learning expert, so you're it. Every chapter sets you a new problem from the field.

All datasets are **synthetic**. They're generated in this notebook so that we know the ground truth and can check your answers.

### 🧭 How each part works
| Symbol | Meaning |
|---|---|
| 📖 | Recap of the chapter's maths, with the equations derived properly |
| 💻 | We build it **from scratch in NumPy** first; PyTorch only appears once you've earned it |
| 🧩 ⭐ / ⭐⭐ / ⭐⭐⭐ | Puzzle: quick / medium / stretch. Fill in the `TODO`, then run the 🧪 check cell |
| 🔑 | Hidden solution cell. Double-click to open it. Have a go first; it's there when you need it |
| 🏆 | Open challenge with bronze / silver / gold targets to aim for |

> **Important:** later cells use the functions you write. If you get stuck, **run the 🔑 solution cell** (you don't have to read it) so the rest of the notebook still works.
{prev}
### 🗺️ This session (≈ 70 min hands-on)
| | | Suggested time |
|---|---|---|
{CONTENTS}

⏱ **Short on time?** Skip the stretch puzzles **{INFO['stretch']}** in the room and finish them afterwards; they're good practice to finish afterwards.

At the end, `progress()` shows which puzzles you've solved and which are left. It's just for you; nothing is recorded or shared.

*Book figures and problem numbers are cited as e.g. "Fig 3.3" and "P3.8".*
""")

code(r'''
#@title 🐾 Setup: run me first! (plotting style, score keeper, referee)
import math, itertools, time, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(precision=4, suppress=True)

# Transition Bio palette
TB = dict(green="#008759", lgreen="#73AE63", blue="#1366AB", lblue="#73B0E6",
          purple="#8F64B7", red="#CF1D10", orange="#EF8E00", grey="#6B6B6B", sand="#E9D8B4")
plt.rcParams.update({
    "figure.dpi": 110, "figure.figsize": (6, 4), "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": plt.cycler(color=[TB["green"], TB["blue"], TB["orange"],
                                         TB["purple"], TB["red"], TB["lblue"], TB["lgreen"]]),
    "font.size": 10.5, "axes.titleweight": "bold"})

# ---------------- progress tracker (private to you; nothing is shared) ----------------
_SOLVED, _PEEKED, _SEEN = set(), set(), set()
SESSION = __SESSION__

def peek(pid):
    _PEEKED.add(pid)

def check(pid, stars, test, ok_msg="", hint=""):
    """Run `test` (a zero-arg function returning True/False) and report the result."""
    _SEEN.add(pid)
    try:
        ok = bool(test())
    except NotImplementedError:
        print(f"🕳️  Puzzle {pid}: still a TODO. Fill it in and re-run (or run the 🔑 cell).")
        return
    except Exception as e:
        print(f"💥 Puzzle {pid} raised {type(e).__name__}: {e}")
        if hint: print("   💡 Hint:", hint)
        return
    if ok:
        _SOLVED.add(pid)
        print(f"✅ Puzzle {pid} solved!  {ok_msg}")
    else:
        print(f"❌ Puzzle {pid}: not quite.", ("💡 " + hint) if hint else "")

def progress():
    own = sorted(_SOLVED - _PEEKED); helped = sorted(_SOLVED & _PEEKED); todo = sorted(_SEEN - _SOLVED)
    print(f"🐾 {len(_SOLVED)} of {len(_SEEN)} puzzles solved "
          f"({len(own)} on your own, {len(helped)} after opening the solution).")
    if todo: print("   Still to do:", ", ".join(todo))

def close(a, b, tol=1e-6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=tol)

def num_grad(fn, params, eps=1e-6):
    """Central finite-difference gradient of scalar fn(params) w.r.t. a 1D array."""
    params = np.array(params, float); g = np.zeros_like(params)
    for i in range(len(params)):
        d = np.zeros_like(params); d[i] = eps
        g[i] = (fn(params + d) - fn(params - d)) / (2 * eps)
    return g

def medal(score, bronze, silver, gold, higher_is_better=False):
    s = score if higher_is_better else -score
    b, si, g = (bronze, silver, gold) if higher_is_better else (-bronze, -silver, -gold)
    if s >= g: return "🥇 GOLD"
    if s >= si: return "🥈 SILVER"
    if s >= b: return "🥉 BRONZE"
    return "🐣 no medal yet: keep digging"

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print("Setup complete 🐾  | numpy", np.__version__, "| torch available:", HAS_TORCH)
'''.replace('__SESSION__', str(S)), form=True)
