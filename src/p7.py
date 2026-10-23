from nbh import md, code, exercise

md(r'''
---
# Part 7 · Gradients and initialization (Chapter 7): 📣 whispers down the burrow
### 📖 Recap: backpropagation = chain rule + caching
**Toy model** (eq 7.5): one hidden unit per layer, with sin, exp and cos as the activations:
$$\mathrm{f}[x,\boldsymbol\phi]=\beta_3+\omega_3\cos\!\Big[\beta_2+\omega_2\exp\big[\beta_1+\omega_1\sin[\beta_0+\omega_0x]\big]\Big],\qquad \ell_i=(\mathrm{f}[x_i,\boldsymbol\phi]-y_i)^2 .$$

**Forward pass** (eq 7.9): compute and *store* $f_0=\beta_0+\omega_0x,\ h_1=\sin f_0,\ f_1=\beta_1+\omega_1h_1,\ h_2=\exp f_1,\ f_2=\beta_2+\omega_2h_2,\ h_3=\cos f_2,\ f_3=\beta_3+\omega_3h_3$.

**Backward pass #1**: work backwards through the intermediate variables. $\frac{\partial\ell}{\partial f_3}=2(f_3-y)$, then $\frac{\partial\ell}{\partial h_3}=\omega_3\frac{\partial\ell}{\partial f_3}$, then $\frac{\partial\ell}{\partial f_2}=-\sin f_2\,\frac{\partial\ell}{\partial h_3}$, and so on.

**Backward pass #2**: the parameter gradients are $\frac{\partial\ell}{\partial\beta_k}=\frac{\partial\ell}{\partial f_k}$ and $\frac{\partial\ell}{\partial\omega_k}=h_k\frac{\partial\ell}{\partial f_k}$, with $h_0=x$.

**Vector form** (eqs 7.24–7.26), for a deep ReLU net:
$$\frac{\partial\ell}{\partial\boldsymbol\beta_k}=\frac{\partial\ell}{\partial\mathbf{f}_k},\qquad \frac{\partial\ell}{\partial\boldsymbol\Omega_k}=\frac{\partial\ell}{\partial\mathbf{f}_k}\mathbf{h}_k^T,\qquad \frac{\partial\ell}{\partial\mathbf{f}_{k-1}}=\mathbb{I}[\mathbf{f}_{k-1}>0]\odot\Big(\boldsymbol\Omega_k^T\frac{\partial\ell}{\partial\mathbf{f}_k}\Big).$$
''')

code(r'''
# 💻 Toy model forward pass
TOY = dict(beta0=1.0, beta1=2.0, beta2=-0.5, beta3=0.4, omega0=0.1, omega1=-0.4, omega2=2.0, omega3=3.0)
x_toy, y_toy = 2.3, 2.0

def toy_forward(x, p):
    f0 = p["beta0"] + p["omega0"] * x; h1 = np.sin(f0)
    f1 = p["beta1"] + p["omega1"] * h1; h2 = np.exp(f1)
    f2 = p["beta2"] + p["omega2"] * h2; h3 = np.cos(f2)
    f3 = p["beta3"] + p["omega3"] * h3
    return dict(f0=f0, h1=h1, f1=f1, h2=h2, f2=f2, h3=h3, f3=f3)

def toy_loss(x, y, p):
    return (toy_forward(x, p)["f3"] - y) ** 2
print("forward values:", {k: round(v, 4) for k, v in toy_forward(x_toy, TOY).items()}, "| loss", round(toy_loss(x_toy, y_toy, TOY), 4))
''')

md(r'''
### 🧩 Puzzle 7.1 ⭐⭐ Backprop the toy model by hand
Fill in the backward pass. Every line is one multiplication by a local derivative.
''')

exercise("7.1",
r'''
def toy_backward(x, y, p):
    v = toy_forward(x, p)
    # Backward pass #1: derivatives w.r.t. intermediate quantities
    dl_df3 = 2 * (v["f3"] - y)
    dl_dh3 = None   # TODO
    dl_df2 = None   # TODO
    dl_dh2 = None   # TODO
    dl_df1 = None   # TODO
    dl_dh1 = None   # TODO
    dl_df0 = None   # TODO
    # Backward pass #2: derivatives w.r.t. parameters
    grads = dict(beta0=None, omega0=None, beta1=None, omega1=None,      # TODO
                 beta2=None, omega2=None, beta3=None, omega3=None)
    return grads
''',
r'''
def _t71():
    g = toy_backward(x_toy, y_toy, TOY)
    if any(v is None for v in g.values()): raise NotImplementedError
    keys = list(TOY)
    def L(vec): return toy_loss(x_toy, y_toy, dict(zip(keys, vec)))
    gn = num_grad(L, [TOY[k] for k in keys])
    bad = [k for k, n in zip(keys, gn) if not np.isclose(g[k], n, rtol=1e-5, atol=1e-6)]
    if bad: print("   mismatch:", bad)
    return not bad
check("7.1", 2, _t71, "Notice how each line reuses the one before it. That reuse is the whole trick.",
      "d sin = cos, d exp = exp (= h2!), d cos = -sin. And dl/domega_k = h_k * dl/df_k, with h_0 = x.")
''',
r'''
def toy_backward(x, y, p):
    v = toy_forward(x, p)
    dl_df3 = 2 * (v["f3"] - y)
    dl_dh3 = p["omega3"] * dl_df3
    dl_df2 = -np.sin(v["f2"]) * dl_dh3
    dl_dh2 = p["omega2"] * dl_df2
    dl_df1 = v["h2"] * dl_dh2            # derivative of exp is exp
    dl_dh1 = p["omega1"] * dl_df1
    dl_df0 = np.cos(v["f0"]) * dl_dh1
    return dict(beta0=dl_df0, omega0=x * dl_df0, beta1=dl_df1, omega1=v["h1"] * dl_df1,
                beta2=dl_df2, omega2=v["h2"] * dl_df2, beta3=dl_df3, omega3=v["h3"] * dl_df3)
''')

md(r'''
### 🧩 Puzzle 7.2 ⭐⭐⭐ Backprop for a deep ReLU network
Use `deep_forward` from Part 4 (re-created in the carry-over cell), which stores `all_f` and `all_h`, and a least squares loss $\ell=\|\mathbf{f}_K-\mathbf{y}\|^2$ for a **single example** (`x` has shape `(Di, 1)`). Implement the three vector equations above. The checker compares against finite differences and, if it's installed, PyTorch autograd.
''')

exercise("7.2",
r'''
def deep_backward(all_f, all_h, omegas, y):
    """Return (dl_dbetas, dl_domegas) as lists matching betas/omegas, for l = ||f_K - y||^2."""
    K = len(omegas) - 1
    dl_dbetas, dl_domegas = [None] * (K + 1), [None] * (K + 1)
    dl_df = 2 * (all_f[-1] - y)
    for k in range(K, -1, -1):
        # TODO: fill dl_dbetas[k], dl_domegas[k]; then (if k > 0) propagate dl_df to layer k-1
        raise NotImplementedError
    return dl_dbetas, dl_domegas
''',
r'''
def _t72():
    rr = np.random.default_rng(0); dims = [3, 5, 4, 2]
    betas, omegas = init_params(dims, rng=rr); betas = [b + rr.normal(0, .3, b.shape) for b in betas]
    xin = rr.normal(size=(3, 1)); yt = rr.normal(size=(2, 1))
    out, fs, hs = deep_forward(xin, betas, omegas); db, dO = deep_backward(fs, hs, omegas, yt)
    def L_w(k, i, j, e):
        o2 = [o.copy() for o in omegas]; o2[k][i, j] += e
        return np.sum((deep_forward(xin, betas, o2)[0] - yt) ** 2)
    def L_b(k, i, e):
        b2 = [b.copy() for b in betas]; b2[k][i, 0] += e
        return np.sum((deep_forward(xin, b2, omegas)[0] - yt) ** 2)
    e = 1e-6; ok = True
    for k in range(len(omegas)):
        gO = np.array([[(L_w(k, i, j, e) - L_w(k, i, j, -e)) / (2 * e) for j in range(omegas[k].shape[1])] for i in range(omegas[k].shape[0])])
        gb = np.array([[(L_b(k, i, e) - L_b(k, i, -e)) / (2 * e)] for i in range(betas[k].shape[0])])
        ok &= close(dO[k], gO, 1e-5) and close(db[k], gb, 1e-5)
    if HAS_TORCH and ok:
        tb = [torch.tensor(b, requires_grad=True) for b in betas]; to = [torch.tensor(o, requires_grad=True) for o in omegas]
        h = torch.tensor(xin)
        for k in range(len(to)):
            f = tb[k] + to[k] @ h; h = torch.relu(f) if k < len(to) - 1 else f
        ((h - torch.tensor(yt)) ** 2).sum().backward()
        ok &= all(close(dO[k], to[k].grad.numpy(), 1e-10) for k in range(len(to)))
        print("   matches torch autograd ✔")
    return ok
check("7.2", 3, _t72, "You just wrote loss.backward(). Seriously, that's all it does (for MLPs).",
      "dl_domegas[k] = dl_df @ all_h[k].T ; dl_df = (all_f[k-1] > 0) * (omegas[k].T @ dl_df)")
''',
r'''
def deep_backward(all_f, all_h, omegas, y):
    K = len(omegas) - 1
    dl_dbetas, dl_domegas = [None] * (K + 1), [None] * (K + 1)
    dl_df = 2 * (all_f[-1] - y)
    for k in range(K, -1, -1):
        dl_dbetas[k] = dl_df.sum(axis=1, keepdims=True)
        dl_domegas[k] = dl_df @ all_h[k].T
        if k > 0:
            dl_df = (all_f[k - 1] > 0) * (omegas[k].T @ dl_df)
    return dl_dbetas, dl_domegas
''')

md(r'''
### 📖 Initialization (§7.5): keep the whisper audible
A message whispered down a 50-meerkat burrow either fades to nothing or turns into a scream. Activations and gradients behave the same way as they pass through layers.

With zero biases and weights $\Omega_{ij}\sim\mathcal{N}(0,\sigma^2_\Omega)$, independent of the activations:
$$\mathbb{E}[f'_i]=0,\qquad \sigma^2_{f'}=\sigma^2_\Omega\sum_{j=1}^{D_h}\mathbb{E}[h_j^2]=\sigma^2_\Omega D_h\,\mathbb{E}[h^2]. \qquad\text{(eq 7.30)}$$
If $f$ is symmetric about zero, then the ReLU gives $\mathbb{E}[h^2]=\tfrac12\sigma_f^2$, so $\sigma^2_{f'}=\tfrac12D_h\sigma^2_\Omega\,\sigma_f^2$ (eq 7.31). Choosing
$$\boxed{\sigma^2_\Omega=\frac{2}{D_h}}\qquad\textbf{(He initialization, eq 7.32)}$$
keeps the variance constant from layer to layer. In the backward pass the same argument uses $D_{h'}$ (eq 7.33), and a compromise between the two is $\sigma^2_\Omega=\frac{4}{D_h+D_{h'}}$ (eq 7.34).
''')

md(r'''
### 🧩 Puzzle 7.3 ⭐⭐ ReLU moments (book P7.14)
For $z\sim\mathcal{N}(0,\sigma^2)$ and $h=\mathrm{ReLU}[z]$, derive $\mathbb{E}[h]$ and $\mathbb{E}[h^2]$ in closed form. (Hint: $\int_0^\infty z\,\mathcal{N}(z;0,\sigma^2)\,dz=\sigma/\sqrt{2\pi}$, and by symmetry half of $\mathbb{E}[z^2]$ comes from $z>0$.)
''')

exercise("7.3",
r'''
def relu_moments(sigma):
    """Return (E[h], E[h^2]) for h = ReLU(z), z ~ N(0, sigma^2)."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t73():
    z = np.random.default_rng(0).normal(0, 1.7, 2_000_000); h = relu(z)
    m1, m2 = relu_moments(1.7); print(f"   MC: E[h]={h.mean():.4f}, E[h²]={np.mean(h**2):.4f}   yours: {m1:.4f}, {m2:.4f}")
    return np.isclose(m1, 1.7 / np.sqrt(2 * np.pi), rtol=1e-9) and np.isclose(m2, 1.7 ** 2 / 2, rtol=1e-9)
check("7.3", 2, _t73, "The factor 1/2 in He init comes straight from E[h^2] = sigma^2 / 2.",
      "E[h] = sigma/sqrt(2*pi), E[h^2] = sigma^2/2")
''',
r'''
def relu_moments(sigma):
    return sigma / np.sqrt(2 * np.pi), sigma ** 2 / 2
''')

code(r'''
# 💻 Recreate Fig 7.7: 50 layers x 100 units, five weight variances
def init_experiment(sigma2, K=50, D=100, seed=0):
    rr = np.random.default_rng(seed)
    betas, omegas = init_params([D] + [D] * K + [1], sigma2=sigma2, rng=rr)
    xin = rr.normal(size=(D, 1))
    out, fs, hs = deep_forward(xin, betas, omegas)
    db, dO = deep_backward(fs, hs, omegas, np.zeros((1, 1)))
    fwd = [np.var(h) for h in hs[1:]]
    bwd = [np.var(g) for g in dO[:-1]]
    return fwd, bwd

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for s2, col in zip([0.001, 0.01, 0.02, 0.1, 1.0], [TB["blue"], TB["lblue"], TB["green"], TB["orange"], TB["red"]]):
    fwd, bwd = init_experiment(s2)
    ax[0].semilogy(fwd, color=col, lw=2, label=fr"$\sigma^2_\Omega$={s2}")
    ax[1].semilogy(bwd, color=col, lw=2, label=fr"$\sigma^2_\Omega$={s2}")
ax[0].set_title("forward: variance of activations"); ax[0].set_xlabel("layer")
ax[1].set_title("backward: variance of weight gradients"); ax[1].set_xlabel("layer")
ax[0].legend(); plt.show()
print("Only σ² = 2/D_h = 0.02 (He) keeps both curves flat. Everything else vanishes or explodes exponentially.")
''')

md(r'''
### 🧩 Puzzle 7.4 ⭐ Predict the explosion
The input $\mathbf{x}$ has $D_i$ standard-normal entries, all biases are zero, every layer has $D_h$ units, and every weight has variance $\sigma^2_\Omega$. Write the variance of the pre-activations $\mathbf{f}_k$ (with $k=0$ being the first layer, which has no ReLU on its input) in closed form. The checker compares it with a simulation.
''')

exercise("7.4",
r'''
def predicted_preact_var(k, sigma2, Di, Dh):
    # TODO
    raise NotImplementedError
''',
r'''
def _t74():
    rr = np.random.default_rng(1); Di = Dh = 200; s2 = 0.013; k = 6
    sims = []
    for _ in range(60):
        b, o = init_params([Di] + [Dh] * 7 + [1], sigma2=s2, rng=rr)
        sims.append(np.var(deep_forward(rr.normal(size=(Di, 1)), b, o)[1][k]))
    pred = predicted_preact_var(k, s2, Di, Dh); print(f"   simulated {np.mean(sims):.4f}  vs  predicted {pred:.4f}")
    return np.isclose(pred, Di * s2 * (Dh * s2 / 2) ** k) and abs(np.mean(sims) / pred - 1) < 0.15
check("7.4", 1, _t74, "A geometric series: stable only when Dh*sigma2/2 = 1.",
      "var(f_0) = Di*sigma2*1; each later layer multiplies by Dh*sigma2/2")
''',
r'''
def predicted_preact_var(k, sigma2, Di, Dh):
    return Di * sigma2 * (Dh * sigma2 / 2) ** k
''')

md(r'''
### 💻 The book's training loop (Fig 7.8), in PyTorch
Everything we built by hand now fits in about 20 lines: He initialization, an MSE loss, SGD with momentum, a step learning-rate schedule, and `loss.backward()` doing Puzzle 7.2 for us.
''')

code(r'''
if HAS_TORCH:
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(0)
    D_i, D_k, D_o = 10, 40, 5
    model = nn.Sequential(nn.Linear(D_i, D_k), nn.ReLU(), nn.Linear(D_k, D_k), nn.ReLU(), nn.Linear(D_k, D_o))
    def weights_init(layer_in):                                # He initialization
        if isinstance(layer_in, nn.Linear):
            nn.init.kaiming_normal_(layer_in.weight); layer_in.bias.data.fill_(0.0)
    model.apply(weights_init)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    x = torch.randn(100, D_i); y = torch.randn(100, D_o)       # random data, as in the book
    data_loader = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=True)
    hist = []
    for epoch in range(100):
        epoch_loss = 0.0
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad(); loss = criterion(model(x_batch), y_batch)
            loss.backward(); optimizer.step(); epoch_loss += loss.item()
        hist.append(epoch_loss); scheduler.step()
    plt.semilogy(hist, color=TB["green"]); plt.xlabel("epoch"); plt.ylabel("training loss")
    plt.title("Fitting 100 random targets (lr 0.01, halved every 10 epochs)"); plt.show()
''')
