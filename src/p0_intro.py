from nbh import md, code

import nbh
S = nbh.SESSION
INFO = {
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
prev = {1: "", 2: "\n> 📦 **Coming from Session 1?** The hidden *carry-over* cell after Setup re-creates everything this session needs from earlier parts, so you don't need to re-run old notebooks.\n",
        3: "\n> 📦 **Coming from Sessions 1–2?** The hidden *carry-over* cell after Setup re-creates everything this session needs from earlier parts (e.g. `deep_forward` from Part 4), so you don't need to re-run old notebooks.\n"}[S]

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
| 🧩 ⭐ / ⭐⭐ / ⭐⭐⭐ | Puzzle worth 1 / 2 / 3 points. Fill in the `TODO`, then run the 🧪 check cell |
| 🔑 | Hidden solution cell. Double-click to open it. **Peeking halves your points**, so be honest! |
| 🏆 | Open challenge with bronze / silver / gold medals |

> **Important:** later cells use the functions you write. If you get stuck, **run the 🔑 solution cell** (you don't have to read it) so the rest of the notebook still works.
{prev}
### 🗺️ This session (≈ 70 min hands-on, {INFO['maxpts']} points available)
| | | Suggested time |
|---|---|---|
{CONTENTS}

⏱ **Short on time?** Skip the stretch puzzles **{INFO['stretch']}** in the room and finish them afterwards; they still count.

At the end, `scoreboard()` gives you your rank in the mob for this session: 🐣 Pup → 🍼 Babysitter → 🦂 Scorpion Hunter → 🔭 Sentinel → 👑 Dominant Female. Note your points; the series total is out of **72**.

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

# ---------------- score keeper ----------------
_SCORE, _PEEKED, _MAXPTS = {}, set(), {}
SESSION, SESSION_MAX = __SESSION__, __MAXPTS__
RANKS = [(0, "🐣 Pup"), (0.15 * SESSION_MAX, "🍼 Babysitter"), (0.4 * SESSION_MAX, "🦂 Scorpion Hunter"),
         (0.65 * SESSION_MAX, "🔭 Sentinel"), (0.85 * SESSION_MAX, "👑 Dominant Female: you run this mob")]

def peek(pid):
    _PEEKED.add(pid)

def check(pid, pts, test, ok_msg="", hint=""):
    """Run `test` (a zero-arg function returning True/False) and award points."""
    _MAXPTS[pid] = pts
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
        got = pts / 2 if pid in _PEEKED else pts
        _SCORE[pid] = max(_SCORE.get(pid, 0), got)
        tag = "  (peeked: half points)" if pid in _PEEKED else ""
        print(f"✅ Puzzle {pid} solved! +{got:g} pt{'s' if got != 1 else ''}{tag}  {ok_msg}")
    else:
        print(f"❌ Puzzle {pid}: not quite.", ("💡 " + hint) if hint else "")

def scoreboard():
    total = sum(_SCORE.values())
    rank = [r for t, r in RANKS if total >= t][-1]
    print(f"🏅 Session {SESSION} score: {total:g} / {SESSION_MAX} points from {len(_SCORE)} puzzles   →   Rank: {rank}")
    return total

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
'''.replace('__SESSION__', str(S)).replace('__MAXPTS__', str(INFO['maxpts'])), form=True)
