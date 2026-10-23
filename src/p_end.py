from nbh import md, code
md(r'''
---
# 🏁 Your progress
''')
code(r'''
progress()
''')
md(r"""
### ⏭️ Next time
**Next up · Chapter 9 (Regularization):** why do over-parameterised networks generalise at all?

### 📚 Where to go next
- The book's own notebooks ([udlbook GitHub](https://github.com/udlbook/udlbook)) mirror every section here. Try Notebooks 3.3 (regions), 6.5 (Adam), 7.2 (backprop) and 8.3 (double descent).
- MNIST-1D (Greydanus, 2020) is the real version of our meerkat call dataset: `pip install mnist1d`.
- Discussion prompts for the group:
  1. A deep network is a flexible classifier. Where would *bias* versus *variance* dominate on real phase-separation data with ~100 measured proteins?
  2. Which output distribution (Part 5) suits a *condensate count per cell*? A *droplet size*? A *partition coefficient*?
  3. Double descent says "bigger can be better". When is that **not** true for our assay data?

*Built for the Transition Bio ML reading group · based on Prince, Understanding Deep Learning (MIT Press, CC-BY-NC-ND) · all data synthetic.*
""")
