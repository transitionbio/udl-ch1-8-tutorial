from nbh import md, code, exercise

md(r'''
---
# Part 4 · Deep neural networks (Chapter 4): 🏜️ folding the dune sea
### 📖 Recap
**Composing** two shallow networks, $x\to y\to y'$, gives a deep network. The first network **folds** the input space onto itself, so several $x$ values map to the same $y$. The second network then applies its function, and that function is **replicated** at every fold (Figs 4.1 and 4.3).

Substituting the first network into the second gives a two-layer network (eqs 4.7–4.9):
$$\mathbf{h}=a[\boldsymbol\theta_0+\boldsymbol\theta x],\quad \mathbf{h}'=a[\boldsymbol\psi_0+\boldsymbol\Psi\mathbf{h}],\quad y'=\phi_0'+\boldsymbol\phi'\mathbf{h}' .$$
In the composed case $\boldsymbol\Psi$ is forced to be an outer product, so a general deep network is *more* expressive than the composition.

**General $K$-layer network** (eq 4.15):
$$\mathbf{h}_1=a[\boldsymbol\beta_0+\boldsymbol\Omega_0\mathbf{x}],\;\; \mathbf{h}_{k+1}=a[\boldsymbol\beta_k+\boldsymbol\Omega_k\mathbf{h}_k],\;\; \mathbf{y}=\boldsymbol\beta_K+\boldsymbol\Omega_K\mathbf{h}_K,$$
with $\boldsymbol\Omega_k\in\mathbb{R}^{D_{k+1}\times D_k}$ and $\boldsymbol\beta_k\in\mathbb{R}^{D_{k+1}}$.

**Shallow vs deep** (1 input, 1 output, $D>2$ units per layer):

| | parameters | max linear regions |
|---|---|---|
| shallow | $3D+1$ | $D+1$ |
| $K$ layers | $3D+1+(K-1)D(D+1)$ | $(D+1)^K$ |
''')

code(r'''
# 💻 Composition = folding.  Net 1 zig-zags [-1,1] -> [-1,1] with three alternating-slope pieces.
def net1(x):
    return shallow_1d(x, -1.0, np.array([3.0, -6.0, 6.0]), np.array([1.0, 1/3, -1/3]), np.ones(3))
def net2(y):   # joints at -0.3 and 0.5 (third joint at -2.0 is outside [-1, 1])
    return shallow_1d(y, -0.5, np.array([2.0, -3.0, 1.0]), np.array([-0.3, -0.5, 2.0]), np.array([-1.0, 1.0, 1.0]))

def count_regions_1d(fn, lo=-1, hi=1, n=400_001, tol=1e-6):
    """Count linear regions of a piecewise-linear fn on [lo, hi] by finding slope changes."""
    x = np.linspace(lo, hi, n); s = np.diff(fn(x)) / np.diff(x)
    flags = np.abs(np.diff(s)) > tol * (1 + np.abs(s).max())
    starts = np.flatnonzero(flags & ~np.concatenate([[False], flags[:-1]]))   # merge neighbours
    return len(starts) + 1

x = np.linspace(-1, 1, 2001)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(x, net1(x), color=TB["blue"], lw=2); ax[0].axhline(0.4, color=TB["orange"], ls=":")
ax[0].set_title(f"net 1 (folds): {count_regions_1d(net1)} regions"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[1].plot(x, net2(x), color=TB["purple"], lw=2); ax[1].set_title(f"net 2: {count_regions_1d(net2)} regions on [-1,1]")
ax[1].set_xlabel("y"); ax[1].set_ylabel("y'")
ax[2].text(0.5, 0.5, "?", fontsize=60, ha="center", va="center", color=TB["grey"], transform=ax[2].transAxes)
ax[2].set_title("net2(net1(x)) = ?"); plt.tight_layout(); plt.show()
print("(the dotted line in panel 1: three different x's land on the same y, so net 2 sees them as one point)")
''')

exercise("4.1",
r'''
# 🧩 Puzzle 4.1 ⭐  Predict BEFORE you run the next line:
#    how many linear regions does net2(net1(x)) have on x in [-1, 1]?
my_prediction = None  # TODO
''',
r'''
def _t41():
    if my_prediction is None: raise NotImplementedError
    truth = count_regions_1d(lambda x: net2(net1(x)))
    plt.figure(figsize=(4.5, 3.2)); plt.plot(x, net2(net1(x)), color=TB["green"], lw=2)
    plt.title(f"composition: {truth} regions"); plt.xlabel("x"); plt.ylabel("y'"); plt.show()
    return my_prediction == truth
check("4.1", 1, _t41, "3 folds × 3 regions = 9. Net 2's function gets copied (and flipped) onto each fold.",
      "Net 1 maps each of its 3 pieces onto the WHOLE of [-1, 1].")
''',
r'''
my_prediction = 9
''')

md(r'''
### 🧩 Puzzle 4.2 ⭐⭐ The general deep network in matrix form (eq 4.15)
Implement `deep_forward` for any depth. `x` has shape `(Di, N)`, so the columns are examples. Return the output **and** lists of the pre-activations $\mathbf{f}_k$ and activations $\mathbf{h}_k$; we'll need them for backprop in Part 7.

Conventions (match the book's eq 7.24):
$\mathbf{f}_0=\boldsymbol\beta_0+\boldsymbol\Omega_0\mathbf{x}$, $\;\mathbf{h}_k=a[\mathbf{f}_{k-1}]$, $\;\mathbf{f}_k=\boldsymbol\beta_k+\boldsymbol\Omega_k\mathbf{h}_k$, and the output is $\mathbf{f}_K$.
''')

code(r'''
# 💻 Random initialisation for any layer sizes (we'll revisit the variance choice in Part 7)
def init_params(dims, sigma2=None, rng=np.random.default_rng(0)):
    """dims = [Di, D1, ..., DK, Do]. Normal weights (He variance 2/D_in by default), zero biases."""
    betas, omegas = [], []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        s2 = 2.0 / d_in if sigma2 is None else sigma2
        omegas.append(rng.normal(0, np.sqrt(s2), (d_out, d_in))); betas.append(np.zeros((d_out, 1)))
    return betas, omegas
''')

exercise("4.2",
r'''
def deep_forward(x, betas, omegas, act=relu):
    """betas[k]: (D_{k+1},1), omegas[k]: (D_{k+1}, D_k). Returns (out, all_f, all_h)
    where all_f = [f_0, ..., f_K] and all_h = [x, h_1, ..., h_K] (all_h[0] is the input)."""
    # TODO
    raise NotImplementedError

''',
r'''
def _t42():
    rr = np.random.default_rng(1)
    th = rr.normal(size=(3, 2)); Ps = rr.normal(size=(3, 4)); ph = rr.normal(size=4)
    betas = [th[:, [0]], Ps[:, [0]], np.array([[ph[0]]])]; omegas = [th[:, [1]], Ps[:, 1:], ph[None, 1:]]
    xx = np.linspace(-2, 2, 50)
    # eq 4.10, written out by hand
    h = relu(th[:, [0]] + th[:, [1]] * xx); hp = relu(Ps[:, [0]] + Ps[:, 1:] @ h); ref = ph[0] + ph[1:] @ hp
    out, fs, hs = deep_forward(xx[None, :], betas, omegas)
    b2, o2 = init_params([3, 4, 2, 3, 2])
    out2 = deep_forward(np.ones((3, 5)), b2, o2)[0]
    return close(out[0], ref, 1e-10) and len(fs) == 3 and len(hs) == 3 and out2.shape == (2, 5)
check("4.2", 2, _t42, "One function for every fully connected net in this notebook.",
      "loop: f = beta + Omega @ h; then h = act(f) for every layer except the last.")
''',
r'''
def deep_forward(x, betas, omegas, act=relu):
    all_f, all_h = [], [x]
    K = len(omegas) - 1
    for k in range(K + 1):
        f = betas[k] + omegas[k] @ all_h[-1]
        all_f.append(f)
        if k < K:
            all_h.append(act(f))
    return all_f[-1], all_f, all_h
''')

md(r'''
### 🧩 Puzzle 4.3 ⭐ Matrix-shape bingo (Fig 4.6)
A network has $D_i=3$ inputs, three hidden layers of sizes $D_1=4$, $D_2=2$, $D_3=3$, and $D_o=2$ outputs. Fill in the shape of every weight matrix and bias vector as a tuple. Write biases as 1-tuples, e.g. `(4,)`.
''')

exercise("4.3",
r'''
shapes = {"Omega0": None, "beta0": None, "Omega1": None, "beta1": None,
          "Omega2": None, "beta2": None, "Omega3": None, "beta3": None}   # TODO
''',
r'''
def _t43():
    if None in shapes.values(): raise NotImplementedError
    ans = {"Omega0": (4, 3), "beta0": (4,), "Omega1": (2, 4), "beta1": (2,),
           "Omega2": (3, 2), "beta2": (3,), "Omega3": (2, 3), "beta3": (2,)}
    wrong = [k for k in ans if tuple(shapes[k]) != ans[k]]
    if wrong: print("   check:", wrong)
    return not wrong
check("4.3", 1, _t43, "Omega_k is (size of layer it feeds) x (size of layer it reads).",
      "Omega_k has shape (D_{k+1}, D_k); beta_k has length D_{k+1}.")
''',
r'''
shapes = {"Omega0": (4, 3), "beta0": (4,), "Omega1": (2, 4), "beta1": (2,),
          "Omega2": (3, 2), "beta2": (3,), "Omega3": (2, 3), "beta3": (2,)}
''')

md(r'''
### 🧩 Puzzle 4.4 ⭐⭐ Regions per parameter (§4.5.2, Fig 4.7a)
Implement the two formulas from the table at the top of this part. Then the cell after the check plots *max regions vs parameters* for $K=1\dots5$. The book's highlighted point is $K=5$, $D=10$: **471 parameters, 161,051 regions**.
''')

exercise("4.4",
r'''
def n_params_deep_1d(D, K):
    # TODO
    raise NotImplementedError

def max_regions_deep_1d(D, K):
    # TODO
    raise NotImplementedError
''',
r'''
def _t44():
    return (n_params_deep_1d(10, 5) == 471 and max_regions_deep_1d(10, 5) == 161051
            and n_params_deep_1d(7, 1) == 22 and max_regions_deep_1d(7, 1) == 8)
check("4.4", 2, _t44, "Depth buys regions much more cheaply than width does.",
      "Each extra layer adds D*(D+1) parameters and multiplies the region count by (D+1).")
''',
r'''
def n_params_deep_1d(D, K):
    return 3 * D + 1 + (K - 1) * D * (D + 1)
def max_regions_deep_1d(D, K):
    return (D + 1) ** K
''')

code(r'''
plt.figure(figsize=(6, 4))
for K in range(1, 6):
    Ds = np.arange(3, 60)
    plt.loglog([n_params_deep_1d(D, K) for D in Ds], [max_regions_deep_1d(D, K) for D in Ds], label=f"K={K}")
plt.plot(471, 161051, "o", color="k", ms=8, mfc="none"); plt.annotate("K=5, D=10", (471, 161051), xytext=(600, 3e3))
plt.xlabel("number of parameters"); plt.ylabel("max linear regions"); plt.legend(); plt.title("Fig 4.7a")
plt.show()
''')

md(r'''
### 🏆 Challenge 4 ⭐⭐⭐ The dune sea: depth efficiency (§4.5.3)
A wind-swept dune field is a **sawtooth**. The **tent layer**
$$t(x)=2\,\mathrm{ReLU}[x]-4\,\mathrm{ReLU}[x-\tfrac12]$$
uses only **2 hidden units** and maps $[0,1]$ onto $[0,1]$ with one fold. Composing it $K$ times gives $2^{K-1}$ dunes, i.e. $2^K$ linear regions, from just $2K$ hidden units.

1. Implement `dune_sea(x, K)` (it applies `tent` K times).
2. A *shallow* net needs one hidden unit per joint. How many hidden units would it need to reproduce `dune_sea(x, 10)` exactly on $[0,1]$?
''')

exercise("C4",
r'''
def tent(x):
    return 2 * relu(x) - 4 * relu(x - 0.5)

def dune_sea(x, K):
    # TODO
    raise NotImplementedError

shallow_units_needed_K10 = None   # TODO
''',
r'''
def _tc4():
    if shallow_units_needed_K10 is None: raise NotImplementedError
    counts = [count_regions_1d(lambda x: dune_sea(x, K), 0, 1, n=200_001) for K in range(1, 7)]
    print("   regions for K=1..6:", counts)
    xs = np.linspace(0, 1, 2000)
    fig, axs = plt.subplots(1, 3, figsize=(12, 2.6), sharey=True)
    for a, K in zip(axs, [1, 3, 6]):
        a.fill_between(xs, dune_sea(xs, K), color=TB["sand"]); a.plot(xs, dune_sea(xs, K), color=TB["orange"])
        a.set_title(f"K={K}: {2*K} hidden units, {2**K} regions")
    plt.show()
    return counts == [2 ** K for K in range(1, 7)] and shallow_units_needed_K10 == 2 ** 10 - 1
check("C4", 3, _tc4, "20 deep units vs 1023 shallow ones: depth efficiency in one picture.",
      "2^K regions means 2^K - 1 joints, and a shallow net needs one unit per joint.")
''',
r'''
def dune_sea(x, K):
    for _ in range(K):
        x = tent(x)
    return x
shallow_units_needed_K10 = 2 ** 10 - 1
''')

code(r'''
# 💻 Reality check: RANDOM nets come nowhere near the maximum region count.
rr = np.random.default_rng(3); res = {}
for K, D in [(1, 30), (2, 6), (4, 3)]:
    cnts = []
    for _ in range(20):
        b, o = init_params([1] + [D] * K + [1], rng=rr)
        b = [bi + rr.normal(0, 0.5, bi.shape) for bi in b]
        cnts.append(count_regions_1d(lambda x: deep_forward(x[None, :], b, o)[0][0], -3, 3, n=100_001))
    res[(K, D)] = (n_params_deep_1d(D, K), np.mean(cnts), max_regions_deep_1d(D, K))
for (K, D), (p, m, mx) in res.items():
    print(f"K={K}, D={D:2d}: {p:3d} params | random init: {m:5.1f} regions on [-3,3] | theoretical max {mx}")
print("\nThe book's caveat (§4.5.2): many regions are possible, but they come with complex dependencies and symmetries.")
''')
