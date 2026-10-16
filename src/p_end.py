import nbh
from nbh import md, code
S = nbh.SESSION
nxt = {0: "**Want more?** The `extended/` folder has the full three-session version: 35 puzzles, every derivation, and a capstone protein-condensate classifier. **Next chapter: 9 (Regularization).**",
       1: "**Session 2 · Losses and training (Ch 5–6):** where do loss functions come from, and how do we walk downhill on a non-convex surface?",
       2: "**Session 3 · Backprop and performance (Ch 7–8):** computing gradients efficiently, initialising deep nets, and whether any of it generalises.",
       3: "**Next up · Chapter 9 (Regularization):** why do over-parameterised networks generalise at all?"}[S]
md(r'''
---
# 🏁 Your progress
''')
code(r'''
progress()
''')
extra = ""
if S in (0, 3):
    extra = r'''
### 📚 Where to go next
- The book's own notebooks ([udlbook GitHub](https://github.com/udlbook/udlbook)) mirror every section here. Try Notebooks 3.3 (regions), 6.5 (Adam), 7.2 (backprop) and 8.3 (double descent).
- MNIST-1D (Greydanus, 2020) is the real version of our meerkat call dataset: `pip install mnist1d`.
- Discussion prompts for the group:
  1. Our condensate model is a flexible classifier. Where would *bias* versus *variance* dominate on real phase-separation data with ~100 measured proteins?
  2. Which output distribution (Part 5) suits a *condensate count per cell*? A *droplet size*? A *partition coefficient*?
  3. Double descent says "bigger can be better". When is that **not** true for our assay data?
'''
md(f"""
### ⏭️ Next time
{nxt}
{extra}
*Built for the Transition Bio ML reading group · based on Prince, Understanding Deep Learning (MIT Press, CC-BY-NC-ND) · all data synthetic.*
""")
