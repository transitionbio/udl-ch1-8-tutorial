from nbh import md, code

ONE_HOUR = r"""
# 🐾 Deep Learning in the Kalahari
## *Understanding Deep Learning*, chapters 1–8 in one hour

**Transition Bio · ML Reading Group**

A one-hour, hands-on tour of chapters 1–8 of Simon J.D. Prince's **_Understanding Deep Learning_** (MIT Press; free at [udlbook.github.io](https://udlbook.github.io/udlbook/)). We follow the book's notation, square brackets and all: $\mathbf{y} = \mathbf{f}[\mathbf{x}, \boldsymbol\phi]$.

### ⏱️ The hour
| Time | What |
|---|---|
| 0–5 min | Slides: welcome and roadmap |
| 5–55 min | Chapter by chapter: a problem slide, then **this notebook** to solve it, then the solution slide |
| 55–60 min | Wrap-up and discussion |

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

⏱ **Short on time?** Skip the 🏆 challenges first. At the end, `progress()` shows what you've solved; it's just for you, and nothing is recorded or shared.

*Book figures and problem numbers are cited as e.g. "Fig 3.3" and "P3.8".*
"""

md(ONE_HOUR)

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
''', form=True)
