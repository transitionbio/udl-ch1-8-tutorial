from nbh import md, code, exercise

md(r'''
---
# Part 8 · Measuring performance (Chapter 8): 🎙️ the meerkat call classifier
### 📖 Recap
- **Generalization** is measured on a separate **test set**. Training error can reach zero while test error stays high.
- For least squares regression, the expected test error splits *exactly* into three parts (eq 8.7):
$$\mathbb{E}_{\mathcal{D}}\Big[\mathbb{E}_y\big[L[x]\big]\Big]=\underbrace{\mathbb{E}_{\mathcal{D}}\Big[\big(\mathrm{f}[x,\boldsymbol\phi[\mathcal{D}]]-\mathrm{f}_\mu[x]\big)^2\Big]}_{\text{variance}}+\underbrace{\big(\mathrm{f}_\mu[x]-\mu[x]\big)^2}_{\text{bias}^2}+\underbrace{\sigma^2}_{\text{noise}}$$
  where $\mathrm{f}_\mu[x]=\mathbb{E}_{\mathcal{D}}\big[\mathrm{f}[x,\boldsymbol\phi[\mathcal{D}]]\big]$ is the average fitted model over training sets.
- **Variance** falls with more data. **Bias** falls with more capacity, which usually raises variance: the **bias–variance trade-off**.
- **Double descent**: as capacity grows past the point where the model can memorise the training data, test error can fall *again*.
- The **curse of dimensionality** means high-dimensional training data are extremely sparse, so behaviour *between* data points (the inductive bias) matters.
- Choose **hyperparameters** on a **validation** set, and touch the test set only once.

#### Derivation sketch (eqs 8.2–8.3)
Add and subtract $\mu[x]$: $\;(\mathrm{f}-y)^2=(\mathrm{f}-\mu)^2+2(\mathrm{f}-\mu)(\mu-y)+(\mu-y)^2$. Taking $\mathbb{E}_y$ kills the cross term because $\mathbb{E}_y[y]=\mu$. Repeat the trick with $\mathrm{f}_\mu$ and $\mathbb{E}_{\mathcal{D}}$ to split $(\mathrm{f}-\mu)^2$ into variance plus bias².
''')

code(r'''
# 💻 The meerkat call dataset: an MNIST-1D-style benchmark (Greydanus 2020), meerkat edition.
CALLS = ["eagle alarm", "jackal alarm", "snake alarm", "close call", "lead call",
         "moving call", "sentinel song", "recruitment", "pup begging", "social chirp"]

def make_meerkat_calls(n, seed=0, noise=0.75, shift=12):
    tr = np.random.default_rng(1234)                             # fixed templates (same species, same repertoire!)
    templates = np.cumsum(tr.normal(0, 1, (10, 12)), axis=1); templates -= templates.mean(1, keepdims=True)
    templates /= np.abs(templates).max(1, keepdims=True)
    r = np.random.default_rng(seed); y = r.integers(0, 10, n); X = np.zeros((n, 40))
    t40 = np.linspace(0, 1, 40)
    for i in range(n):
        tpl = templates[y[i]]; stretch = r.uniform(0.75, 1.25)
        tt = np.clip((t40 - 0.5) * stretch + 0.5, 0, 1)
        sig = np.interp(tt, np.linspace(0, 1, 12), tpl) * r.uniform(0.7, 1.3)
        sig = np.roll(sig, r.integers(-shift, shift + 1))
        sig += r.uniform(-0.6, 0.6) * (t40 - 0.5)                                 # slope (background wind)
        sig += noise * np.convolve(r.normal(size=44), np.ones(5) / 5, "valid")    # correlated noise
        X[i] = sig + 0.15 * noise * r.normal(size=40)
    return X.astype(np.float32), y

Xtr, ytr = make_meerkat_calls(4000, seed=1); Xte, yte = make_meerkat_calls(1000, seed=2)
fig, axs = plt.subplots(2, 5, figsize=(13, 4), sharex=True, sharey=True)
for k, ax in enumerate(axs.ravel()):
    for j in np.flatnonzero(ytr == k)[:4]: ax.plot(Xtr[j], lw=1)
    ax.set_title(CALLS[k], fontsize=9)
plt.suptitle("Four examples of each call type (40 samples each)", fontweight="bold"); plt.tight_layout(); plt.show()
''')

code(r'''
# 💻 Train the §8.1 model: 40 → 100 → 100 → 10, SGD, lr 0.1, batch 100, 6000 steps
def train_mlp(Xtr, ytr, Xev, yev, width=100, depth=2, lr=0.1, steps=6000, batch=100, seed=0, opt="sgd", log_every=100):
    torch.manual_seed(seed)
    layers, d = [], Xtr.shape[1]
    for _ in range(depth): layers += [torch.nn.Linear(d, width), torch.nn.ReLU()]; d = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 10))
    for m in net:
        if isinstance(m, torch.nn.Linear): torch.nn.init.kaiming_normal_(m.weight); m.bias.data.zero_()
    o = torch.optim.SGD(net.parameters(), lr=lr) if opt == "sgd" else torch.optim.Adam(net.parameters(), lr=lr)
    Xt, yt, Xe, ye = map(torch.tensor, (Xtr, ytr, Xev, yev)); lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed); hist = []
    for s in range(steps + 1):
        if s % log_every == 0:
            with torch.no_grad():
                ot, oe = net(Xt), net(Xe)
                hist.append((s, (ot.argmax(1) != yt).float().mean().item(), (oe.argmax(1) != ye).float().mean().item(),
                             lossf(ot, yt).item(), lossf(oe, ye).item()))
        if s == steps: break
        idx = torch.randint(0, len(Xt), (batch,), generator=g)
        o.zero_grad(); lossf(net(Xt[idx]), yt[idx]).backward(); o.step()
    return net, np.array(hist)

if HAS_TORCH:
    t0 = time.time(); _, H = train_mlp(Xtr, ytr, Xte, yte); print(f"trained in {time.time()-t0:.1f}s")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(H[:, 0], 100 * H[:, 1], label="train", color=TB["green"]); ax[0].plot(H[:, 0], 100 * H[:, 2], label="test", color=TB["orange"])
    ax[0].set_ylabel("% error"); ax[0].set_xlabel("step"); ax[0].legend(); ax[0].set_title("Error (cf. Fig 8.2a)")
    ax[1].plot(H[:, 0], H[:, 3], label="train", color=TB["green"]); ax[1].plot(H[:, 0], H[:, 4], label="test", color=TB["orange"])
    ax[1].set_xlabel("step"); ax[1].legend(); ax[1].set_title("Loss (cf. Fig 8.2b)"); plt.show()
    print(f"final train error {100*H[-1,1]:.1f}%   test error {100*H[-1,2]:.1f}%   (chance = 90%)")
''')

exercise("8.1",
r'''
# 🧩 Puzzle 8.1 ⭐  The test LOSS climbs late in training while the test ERROR stays flat. Why?
#  (a) the learning rate is too high, so the optimiser diverges
#  (b) the model keeps making the same mistakes but with growing confidence (softmax logits pushed to extremes)
#  (c) the test set is mislabelled
#  (d) ReLUs die during training
answer_8_1 = None   # TODO "a" / "b" / "c" / "d"
''',
r'''
def _t81():
    if answer_8_1 is None: raise NotImplementedError
    return answer_8_1.strip().lower() == "b"
check("8.1", 1, _t81, "Confidently wrong predictions get punished by the log in the cross-entropy (see Part 5B).")
''',
r'''
answer_8_1 = "b"
''')

md(r'''
### 📖 Noise, bias and variance, measured exactly (§8.2)
To *measure* the three terms we need to know the truth, so we go back to 1D regression. The true function is $\mu[x]=\exp(\sin(2\pi x))$, the noise has $\sigma=0.3$, and each dataset has 15 points (one per equal segment of $[0,1]$). The **simplified network** of Fig 8.4 fixes the joints at $0,\frac1D,\dots,\frac{D-1}{D}$, so only the output layer is learned. That makes it **linear in its parameters**, so we can fit it in closed form (book P8.3):
$$\mathbf{A}=\big[\mathbf{1},\ \mathrm{ReLU}[\mathbf{x}-0],\ \mathrm{ReLU}[\mathbf{x}-\tfrac1D],\dots\big],\qquad \hat{\boldsymbol\phi}=(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{y}.$$
''')

code(r'''
def mu_true(x): return np.exp(np.sin(2 * np.pi * x))
SIGMA = 0.3
def make_reg_data(n, r):
    x = (np.arange(n) + r.uniform(size=n)) / n          # one sample per equal segment
    return x, mu_true(x) + r.normal(0, SIGMA, n)
''')

exercise("8.2",
r'''
def design(x, D):
    """[1, ReLU(x - 0/D), ReLU(x - 1/D), ..., ReLU(x - (D-1)/D)] -> shape (len(x), D+1)"""
    # TODO
    raise NotImplementedError

def fit_fixed_joints(x, y, D):
    """Least squares coefficients (D+1,). Use np.linalg.lstsq for numerical safety."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t82():
    r = np.random.default_rng(0); x, y = make_reg_data(15, r)
    A = design(x, 3); c = fit_fixed_joints(x, y, 3)
    ref = np.column_stack([np.ones(15)] + [np.maximum(x - k / 3, 0) for k in range(3)])
    return A.shape == (15, 4) and close(A, ref) and close(c, np.linalg.lstsq(ref, y, rcond=None)[0], 1e-8)
check("8.2", 2, _t82, "Fixing the hidden layer makes the problem linear in its parameters, so it's convex and solvable in closed form.",
      "np.column_stack([np.ones_like(x)] + [relu(x - k/D) for k in range(D)])")
''',
r'''
def design(x, D):
    return np.column_stack([np.ones_like(x)] + [relu(x - k / D) for k in range(D)])
def fit_fixed_joints(x, y, D):
    return np.linalg.lstsq(design(x, D), y, rcond=None)[0]
''')

md(r'''
### 🧩 Puzzle 8.3 ⭐⭐⭐ Estimate bias² and variance by brute force
For a given $D$: draw `n_datasets` training sets, fit each one, and evaluate the fits on a grid `xg = np.linspace(0, 1, 200)`.
- $\mathrm{f}_\mu$ = the mean of the fits over datasets.
- **variance** = the variance across datasets, averaged over the grid.
- **bias²** = $(\mathrm{f}_\mu-\mu)^2$, averaged over the grid.

The checker draws *fresh* test data and verifies eq 8.7: **bias² + variance + σ² ≈ measured test MSE**.
''')

exercise("8.3",
r'''
def bias_variance(D, n_datasets=300, n_points=15, seed=0):
    """Return (bias2, variance), each averaged over xg = np.linspace(0, 1, 200)."""
    r = np.random.default_rng(seed)
    # TODO
    raise NotImplementedError
''',
r'''
def _t83():
    D = 3; b2, var = bias_variance(D, 400, 15, seed=1)
    r = np.random.default_rng(99); errs = []
    for _ in range(400):
        x, y = make_reg_data(15, r); c = fit_fixed_joints(x, y, D)
        xt, yt = r.uniform(size=200), None; yt = mu_true(xt) + r.normal(0, SIGMA, 200)
        errs.append(np.mean((design(xt, D) @ c - yt) ** 2))
    pred = b2 + var + SIGMA ** 2; meas = np.mean(errs)
    print(f"   bias²={b2:.4f}  var={var:.4f}  noise={SIGMA**2:.4f}  →  sum={pred:.4f}   measured test MSE={meas:.4f}")
    return abs(pred / meas - 1) < 0.06
check("8.3", 3, _t83, "Eq 8.7, confirmed by simulation.",
      "fits = np.array([design(xg, D) @ fit_fixed_joints(*make_reg_data(n_points, r), D) for _ in range(n_datasets)])")
''',
r'''
def bias_variance(D, n_datasets=300, n_points=15, seed=0):
    r = np.random.default_rng(seed); xg = np.linspace(0, 1, 200)
    fits = np.array([design(xg, D) @ fit_fixed_joints(*make_reg_data(n_points, r), D) for _ in range(n_datasets)])
    f_mu = fits.mean(0)
    return np.mean((f_mu - mu_true(xg)) ** 2), np.mean(fits.var(0))
''')

code(r'''
# 💻 The bias-variance trade-off (Fig 8.9)
Ds = np.arange(1, 13); bv = np.array([bias_variance(D, 300, 15) for D in Ds])
plt.plot(Ds, bv[:, 0], "o-", color=TB["orange"], label="bias²"); plt.plot(Ds, bv[:, 1], "o-", color=TB["blue"], label="variance")
plt.plot(Ds, bv.sum(1), "--", color=TB["grey"], label="bias² + variance")
best = Ds[np.argmin(bv.sum(1))]; plt.axvline(best, color=TB["green"], ls=":", label=f"best capacity D={best}")
plt.yscale("log"); plt.xlabel("hidden units D (= linear regions)"); plt.legend(); plt.title("Bias-variance trade-off, 15 points"); plt.show()
''')

md(r'''
### 📖 Double descent (§8.4)
Below we use a shallow net with a **random, frozen** first layer (random ReLU features) and fit the output weights with the **minimum-norm** least squares solution (`np.linalg.lstsq`). Minimum norm is a simple stand-in for the implicit regularisation the book suspects SGD provides. We have 30 noisy training points and sweep the width from 1 to 2,000 hidden units, tracking both training and test error. We call the width where training MSE first drops below $\sigma^2/10$ the **(near-)interpolation threshold**: past it, the model has essentially memorised the noise.
''')

code(r'''
def random_feature_curve(widths, n_train=30, n_test=500, n_rep=40, seed=0):
    r = np.random.default_rng(seed); te = np.zeros((n_rep, len(widths))); tr = np.zeros_like(te)
    for rep in range(n_rep):
        xtr, ytr_ = make_reg_data(n_train, r); xts = r.uniform(size=n_test); yts = mu_true(xts) + r.normal(0, SIGMA, n_test)
        c = r.uniform(-0.1, 1.1, 2000)                                        # joint positions
        a = np.abs(r.normal(0, 1, 2000)) * r.choice([-1, 1], 2000); b = -a * c # frozen first layer
        for j, D in enumerate(widths):
            feats = lambda x: np.column_stack([np.ones_like(x), relu(a[:D] * x[:, None] + b[:D])])
            w = np.linalg.lstsq(feats(xtr), ytr_, rcond=None)[0]               # minimum-norm when under-determined
            te[rep, j] = np.mean((feats(xts) @ w - yts) ** 2); tr[rep, j] = np.mean((feats(xtr) @ w - ytr_) ** 2)
    return np.median(tr, 0), np.median(te, 0)

widths = np.unique(np.round(np.logspace(0, np.log10(2000), 45)).astype(int))
tr_err, te_err = random_feature_curve(widths)
D_interp = widths[np.argmax(tr_err < 0.1 * SIGMA ** 2)]       # first width that (almost) memorises the training set
plt.loglog(widths, te_err, "o-", color=TB["purple"], ms=4, label="test")
plt.loglog(widths, np.maximum(tr_err, 1e-4), "o-", color=TB["green"], ms=3, label="train (floored at 1e-4)")
plt.axvline(D_interp, color=TB["grey"], ls=":"); plt.text(D_interp * 1.15, 2e-3, "(near-)interpolation\nthreshold", color=TB["grey"])
plt.axhline(SIGMA ** 2, color=TB["orange"], ls="--", lw=1); plt.text(150, SIGMA ** 2 * 0.62, "noise floor σ²", color=TB["orange"])
plt.xlabel("hidden units D"); plt.ylabel("median MSE"); plt.legend(loc="lower left"); plt.title("Double descent with random ReLU features"); plt.show()
print(f"training MSE first drops below σ²/10 at D = {D_interp} | test-error peak at D = {widths[np.argmax(te_err)]}")
print(f"classical sweet spot: {te_err[widths < D_interp].min():.3f} | peak: {te_err.max():.2f} | D=2000: {te_err[-1]:.3f}")
print("In this tiny 1D problem the second descent recovers but doesn't beat the classical sweet spot.\n"
      "On MNIST-1D with label noise (Fig 8.10b) it does. How much it helps depends on data, noise and inductive bias.")
''')

exercise("8.4",
r'''
# 🧩 Puzzle 8.4 ⭐  Look at the plot. The test-error peak sits...
#  (a) at D = 1, where the model is too simple
#  (b) around the width where the model can first (almost) memorise the training data (the interpolation threshold)
#  (c) at the largest width, where there are the most parameters
#  (d) at D = 30 exactly, because there are 30 data points (no more, no less)
answer_8_4 = None   # TODO
''',
r'''
def _t84():
    if answer_8_4 is None: raise NotImplementedError
    return answer_8_4.strip().lower() == "b"
check("8.4", 1, _t84, "Not exactly 30 here: random joints rarely land in every gap between data points, so the net needs more units than data points to memorise them.",
      "Compare the dotted line with the purple peak.")
''',
r'''
answer_8_4 = "b"
''')

md(r'''
### 🧩 Puzzle 8.5 ⭐⭐ The curse of dimensionality (§8.4.1)
1. Quantise each of the call dataset's 40 input dimensions into 10 bins. How many **bins per training example** are there with 10,000 examples? (The book's number.)
2. A hypersphere of diameter 1 sits inside the unit hypercube. Implement `sphere_fraction(d)`, the fraction of the cube's volume inside the sphere: $V_d(r)=\dfrac{\pi^{d/2}}{\Gamma(d/2+1)}r^d$ with $r=\tfrac12$.
''')

exercise("8.5",
r'''
bins_per_example = None     # TODO

def sphere_fraction(d):
    # TODO (math.gamma)
    raise NotImplementedError
''',
r'''
def _t85():
    if bins_per_example is None: raise NotImplementedError
    ok = np.isclose(bins_per_example, 1e36) and np.isclose(sphere_fraction(2), np.pi / 4) and np.isclose(sphere_fraction(3), np.pi / 6)
    print(f"   fraction inside the sphere: d=2 {sphere_fraction(2):.3f}, d=10 {sphere_fraction(10):.4f}, d=40 {sphere_fraction(40):.2e}")
    return ok
check("8.5", 2, _t85, "In 40-D, essentially ALL the volume is in the corners. Your intuition from 3-D does not apply.",
      "10**40 / 10**4; math.pi**(d/2) / math.gamma(d/2 + 1) * 0.5**d")
''',
r'''
bins_per_example = 10 ** 40 / 10 ** 4
def sphere_fraction(d):
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1) * 0.5 ** d
''')

code(r'''
# 💻 Distances concentrate in high dimensions: nearest and farthest neighbours look alike
r = np.random.default_rng(0); dims = [1, 2, 5, 10, 40, 100, 1000]; ratios = []
for d in dims:
    P = r.uniform(size=(1000, d)); q = r.uniform(size=d); dist = np.linalg.norm(P - q, axis=1)
    ratios.append(dist.max() / dist.min())
plt.semilogx(dims, ratios, "o-", color=TB["blue"]); plt.yscale("log")
plt.xlabel("dimension"); plt.ylabel("farthest / nearest distance"); plt.title("Everything is far from everything"); plt.show()
''')

md(r'''
### 🏆 Challenge 8 ⭐⭐⭐ The Meerkat Call Classifier Cup (§8.5: choosing hyperparameters)
Split the 4,000 training calls into **3,000 train / 1,000 validation**, choose hyperparameters using the **validation** error *only*, and then look at the test set **once**. Medals are awarded on validation accuracy:
- 🥉 > 50%
- 🥈 > 54%
- 🥇 > 56.5%

(The out-of-the-box §8.1 model scores about 53%. A noisy 10-class problem with only 3,000 examples is *hard*.)

Things to play with: `width`, `depth`, `lr`, `steps`, `batch`, and `opt` (`"sgd"` or `"adam"`). Regularisation (chapter 9) is off-limits, though early stopping is *kind of* allowed: `steps` is a hyperparameter!
''')

exercise("C8",
r'''
Xa, ya, Xv, yv = Xtr[:3000], ytr[:3000], Xtr[3000:], ytr[3000:]
my_hparams = dict(width=100, depth=2, lr=0.1, steps=2000, batch=100, opt="sgd")   # TODO: search!
''',
r'''
def _tc8():
    if not HAS_TORCH: print("   (torch missing)"); return False
    net, H = train_mlp(Xa, ya, Xv, yv, log_every=max(1, my_hparams["steps"]), **my_hparams)
    acc = 1 - H[-1, 2]; print(f"   validation accuracy {100*acc:.1f}% → {medal(acc, .50, .54, .565, higher_is_better=True)}")
    with torch.no_grad(): test_acc = (net(torch.tensor(Xte)).argmax(1).numpy() == yte).mean()
    print(f"   🔒 test accuracy (look once!): {100*test_acc:.1f}%")
    return acc > 0.565
check("C8", 3, _tc8, "Gold! The station will use your classifier on next season's recordings.",
      "Try Adam (lr ~1e-3), width 256-512, depth 2-3 and ~1000 steps. Watch how val error moves with steps.")
''',
r'''
# A quick validation sweep (honest: only validation numbers are used to choose)
best = None
for width, depth, lr, steps in [(256, 2, 1e-3, 1500), (512, 3, 1e-3, 1000)]:
    hp = dict(width=width, depth=depth, lr=lr, steps=steps, batch=100, opt="adam")
    _, H = train_mlp(Xa, ya, Xv, yv, log_every=steps, **hp); acc = 1 - H[-1, 2]
    print(hp, f"val acc {100*acc:.1f}%")
    if best is None or acc > best[0]: best = (acc, hp)
my_hparams = best[1]
''')
