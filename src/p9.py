from nbh import md, code, exercise

md(r'''
---
# Part 9 · Capstone: 🧪 from meerkats to molecules
Time to put chapters 2–8 together on a problem closer to home. This is a **toy, synthetic** phase-separation dataset, invented for this exercise and *not* TBio data. Each sample is a protein condition with three features:
- $x_1$ = log₁₀ protein concentration (µM)
- $x_2$ = salt (mM, scaled)
- $x_3$ = a made-up "stickiness" score of the sequence

The label is $y=1$ if droplets formed. Following the usual cartoon of a phase diagram, the saturation concentration rises with salt and falls with stickiness. Some labels are flipped, because the imaging is noisy.

**Your job, using every chapter:**

| step | chapter |
|---|---|
| pick the output distribution, final activation and loss | 5 |
| build a deep ReLU network in PyTorch | 3–4 |
| use He initialization | 7 |
| train with Adam on minibatches | 6 |
| tune on validation, report test once | 8 |

### 🏆 Puzzle 9 ⭐⭐⭐: reach ≥ 88% validation accuracy, then plot the learned phase diagram.
''')

code(r'''
def make_condensates(n, seed):
    r = np.random.default_rng(seed)
    logc = r.uniform(-1, 2.5, n); salt = r.uniform(0, 1, n); stick = r.uniform(0, 1, n)
    csat = 0.2 + 1.6 * salt ** 1.5 - 1.1 * stick + 0.4 * np.sin(3 * stick * salt)      # log10 c_sat
    y = (logc > csat).astype(np.float32)
    flip = r.uniform(size=n) < 0.06; y[flip] = 1 - y[flip]
    return np.stack([logc, salt, stick], 1).astype(np.float32), y

Xc_tr, yc_tr = make_condensates(3000, 1); Xc_va, yc_va = make_condensates(1000, 2); Xc_te, yc_te = make_condensates(1000, 3)
fig, ax = plt.subplots(figsize=(5.5, 4))
m = Xc_tr[:, 2] > 0.6
ax.scatter(Xc_tr[m, 1], Xc_tr[m, 0], c=yc_tr[m], cmap=ListedColormap([TB["lblue"], TB["green"]]), s=8)
ax.set_xlabel("salt (scaled)"); ax.set_ylabel(r"log$_{10}$ [protein] (µM)"); ax.set_title("Sticky proteins: green = droplets"); plt.show()
''')

exercise("9",
r'''
def train_condensate_model(Xtr, ytr, Xva, yva):
    """Build, initialise, and train a model. Return a function predict_proba(X_numpy) -> probabilities (numpy)."""
    # TODO: torch.nn.Sequential(...), kaiming init, BCEWithLogitsLoss, Adam, minibatches
    raise NotImplementedError
''',
r'''
def _t9():
    if not HAS_TORCH: print("   (torch missing)"); return False
    predict_proba = train_condensate_model(Xc_tr, yc_tr, Xc_va, yc_va)
    va = np.mean((predict_proba(Xc_va) > 0.5) == yc_va); te = np.mean((predict_proba(Xc_te) > 0.5) == yc_te)
    print(f"   validation accuracy {100*va:.1f}%   |   🔒 test (once) {100*te:.1f}%   |   label-noise ceiling ≈ 94%")
    S, C = np.meshgrid(np.linspace(0, 1, 200), np.linspace(-1, 2.5, 200))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for a, st in zip(axs, [0.1, 0.5, 0.9]):
        grid = np.stack([C.ravel(), S.ravel(), np.full(C.size, st)], 1).astype(np.float32)
        im = a.contourf(S, C, predict_proba(grid).reshape(S.shape), 20, cmap="BuGn", vmin=0, vmax=1)
        a.set_title(f"stickiness = {st}"); a.set_xlabel("salt")
    axs[0].set_ylabel(r"log$_{10}$ [protein]"); plt.colorbar(im, ax=axs, label="P(droplets)"); plt.show()
    return va >= 0.88
check("9", 3, _t9, "A learned phase diagram. Welcome to the intersection of meerkats and molecules.",
      "Standardise inputs, 2-3 hidden layers of 64, BCEWithLogitsLoss, Adam lr 1e-3 to 3e-3, ~30 epochs of batch 64.")
''',
r'''
def train_condensate_model(Xtr, ytr, Xva, yva):
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    for m in net:
        if isinstance(m, torch.nn.Linear): torch.nn.init.kaiming_normal_(m.weight); m.bias.data.zero_()
    X = torch.tensor((Xtr - mu) / sd); Y = torch.tensor(ytr)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3); lossf = torch.nn.BCEWithLogitsLoss()
    for epoch in range(40):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); lossf(net(X[idx])[:, 0], Y[idx]).backward(); opt.step()
    def predict_proba(Xn):
        with torch.no_grad(): return torch.sigmoid(net(torch.tensor((Xn - mu) / sd))[:, 0]).numpy()
    return predict_proba
''')

