from nbh import md, code, exercise

md(r'''
---
# Part 5 · Loss functions (Chapter 5): 🔭 sentinels, alarm calls, scorpions and lost burrows
### 📖 The recipe (§5.2)
Instead of predicting $\mathbf{y}$ directly, the network predicts the **parameters $\boldsymbol\theta$ of a distribution** over $\mathbf{y}$.
1. Choose a distribution $Pr(\mathbf{y}\,|\,\boldsymbol\theta)$ defined on the output domain.
2. Let the network predict the parameters: $\boldsymbol\theta=\mathbf{f}[\mathbf{x},\boldsymbol\phi]$.
3. Train by **maximum likelihood**. Because $\log$ is monotonic and we prefer minimizing, minimize the **negative log-likelihood**:
$$\hat{\boldsymbol\phi}=\underset{\boldsymbol\phi}{\mathrm{argmin}}\Big[-\sum_{i=1}^{I}\log Pr\big(\mathbf{y}_i\,|\,\mathbf{f}[\mathbf{x}_i,\boldsymbol\phi]\big)\Big].$$
4. **Inference**: return the full distribution, or its maximum $\hat{\mathbf{y}}=\mathrm{argmax}_{\mathbf{y}}\,Pr(\mathbf{y}\,|\,\mathbf{f}[\mathbf{x},\hat{\boldsymbol\phi}])$.

| output domain | distribution | final activation | loss |
|---|---|---|---|
| $y\in\mathbb{R}$ | normal | none | least squares |
| $y\in\{0,1\}$ | Bernoulli | logistic sigmoid | binary cross-entropy |
| $y\in\{1..K\}$ | categorical | softmax | multiclass cross-entropy |
| $y\in\{0,1,2,\dots\}$ | Poisson | $\exp$ | Poisson NLL |
| $y\in(-\pi,\pi]$ | von Mises | none | $-\kappa\cos(y-\mu)$ |
''')

md(r'''
## 5A · Univariate regression: least squares is a *derived* result
With $Pr(y|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\big[-\frac{(y-\mu)^2}{2\sigma^2}\big]$ and $\mu=\mathrm{f}[x,\boldsymbol\phi]$:
$$L[\boldsymbol\phi]=-\sum_i\log Pr(y_i|\mathrm{f}[x_i,\boldsymbol\phi],\sigma^2)=\frac{I}{2}\log(2\pi\sigma^2)+\sum_i\frac{(y_i-\mathrm{f}[x_i,\boldsymbol\phi])^2}{2\sigma^2}.$$
The first term doesn't depend on $\boldsymbol\phi$, and $2\sigma^2$ is a positive constant, so the argmin is exactly the **least squares** solution (eq 5.11). The Gaussian assumption is what justifies the loss we used in Part 2.

### 🧩 Puzzle 5.1 ⭐⭐ Normal NLL and the MLE of the variance (§5.3.3)
1. Implement `normal_nll(y, mu, sigma2)` (a sum over points).
2. Differentiate $L$ with respect to $\sigma^2$, set the result to zero, and implement `sigma2_mle(y, mu)`.
''')

exercise("5.1",
r'''
def normal_nll(y, mu, sigma2):
    # TODO
    raise NotImplementedError

def sigma2_mle(y, mu):
    # TODO
    raise NotImplementedError
''',
r'''
def _t51():
    rr = np.random.default_rng(0); y = rr.normal(3, 2, 200); mu = np.full(200, 3.1)
    ref = np.sum(0.5 * np.log(2 * np.pi * 1.7) + (y - mu) ** 2 / (2 * 1.7))
    grid = np.linspace(0.5, 10, 20001); best = grid[np.argmin([normal_nll(y, mu, s) for s in grid])]
    phis = np.linspace(-1, 6, 701)
    same_argmin = all(phis[np.argmin([normal_nll(gain, f_linear(hours, 1.0, p), s2) for p in phis])] ==
                      phis[np.argmin([loss_ls(1.0, p, hours, gain) for p in phis])] for s2 in [0.3, 4, 50])
    return np.isclose(normal_nll(y, mu, 1.7), ref) and abs(sigma2_mle(y, mu) - best) < 1e-3 and same_argmin
check("5.1", 2, _t51, "sigma^2_hat = mean squared residual. Changing sigma never moves the best phi.",
      "dL/dsigma2 = I/(2 sigma2) - sum r^2 / (2 sigma2^2) = 0")
''',
r'''
def normal_nll(y, mu, sigma2):
    return np.sum(0.5 * np.log(2 * np.pi * sigma2) + (y - mu) ** 2 / (2 * sigma2))
def sigma2_mle(y, mu):
    return np.mean((y - mu) ** 2)
''')

md(r'''
### 📖 Heteroscedastic regression (§5.3.4): pups get *more variable* as they grow
Give the network **two outputs**: a mean $\mu=\mathrm{f}_1[x,\boldsymbol\phi]$ and a variance $\sigma^2=\mathrm{softplus}[\mathrm{f}_2[x,\boldsymbol\phi]]$, which keeps the variance positive. Minimize
$$L=\sum_i\Big[\tfrac12\log(2\pi\sigma_i^2)+\frac{(y_i-\mu_i)^2}{2\sigma_i^2}\Big].$$

### 🧩 Puzzle 5.2 ⭐⭐ Heteroscedastic NLL in PyTorch
Write the loss as a torch function, averaged over the batch. A small network then learns how pup mass varies with age, including the growing spread.
''')

exercise("5.2",
r'''
def hetero_nll(y, out):
    """y: (N,), out: (N, 2) raw network outputs [mu, raw_var]. sigma2 = softplus(raw_var) + 1e-6.
    Return the MEAN negative log-likelihood (a scalar tensor)."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t52():
    if not HAS_TORCH: print("   (torch missing, skipping)"); return True
    out = torch.tensor([[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]]); y = torch.tensor([1.5, 1.0, -0.5])
    s2 = torch.nn.functional.softplus(out[:, 1]) + 1e-6
    ref = (0.5 * torch.log(2 * math.pi * s2) + (y - out[:, 0]) ** 2 / (2 * s2)).mean()
    return torch.allclose(hetero_nll(y, out), ref)
check("5.2", 2, _t52, "Now the network can say how sure it is.",
      "torch.nn.functional.softplus(out[:,1]) + 1e-6")
''',
r'''
def hetero_nll(y, out):
    mu = out[:, 0]; s2 = torch.nn.functional.softplus(out[:, 1]) + 1e-6
    return (0.5 * torch.log(2 * math.pi * s2) + (y - mu) ** 2 / (2 * s2)).mean()
''')

code(r'''
# 💻 Train the heteroscedastic pup-growth model
if HAS_TORCH:
    torch.manual_seed(0); rr = np.random.default_rng(4)
    age = rr.uniform(0, 1, 400)                                     # age (scaled 0..1 ≈ 0..120 days)
    mass = 30 + 220 * (1 - np.exp(-2.5 * age)) + rr.normal(0, 4 + 30 * age)   # noise grows with age
    X = torch.tensor(age, dtype=torch.float32)[:, None]; Y = torch.tensor((mass - 150) / 50, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
                              torch.nn.ReLU(), torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for step in range(2000):
        opt.zero_grad(); loss = hetero_nll(Y, net(X)); loss.backward(); opt.step()
    with torch.no_grad():
        g = torch.linspace(0, 1, 200)[:, None]; o = net(g)
        mu = o[:, 0].numpy() * 50 + 150; sd = np.sqrt(torch.nn.functional.softplus(o[:, 1]).numpy() + 1e-6) * 50
    plt.scatter(age * 120, mass, s=6, color=TB["orange"], alpha=0.6, label="pups")
    plt.plot(g[:, 0] * 120, mu, color=TB["green"], lw=2, label=r"$\mu$")
    plt.fill_between(g[:, 0].numpy() * 120, mu - 2 * sd, mu + 2 * sd, color=TB["lgreen"], alpha=0.3, label=r"$\mu\pm2\sigma$")
    plt.xlabel("age (days)"); plt.ylabel("mass (g)"); plt.legend(); plt.title("Heteroscedastic regression"); plt.show()
''')

md(r'''
## 5B · Binary classification: 🔭 will this meerkat go on sentinel duty?
Meerkats tend to stand guard once they've **eaten well**: a full stomach makes guarding cheap. Our feature is $x$ = grams of food eaten in the last hour, and the label is $y=1$ if the animal goes on guard.

- **Bernoulli**: $Pr(y|\lambda)=(1-\lambda)^{1-y}\lambda^{y}$, with $\lambda=\mathrm{sig}[\mathrm{f}[x,\boldsymbol\phi]]$ and $\mathrm{sig}[z]=\frac{1}{1+e^{-z}}$.
- **Binary cross-entropy** (eq 5.20): $L=\sum_i -(1-y_i)\log\big(1-\mathrm{sig}[\mathrm{f}_i]\big)-y_i\log\mathrm{sig}[\mathrm{f}_i]$.

### 🧩 Puzzle 5.3 ⭐⭐ The prettiest derivative in deep learning
Show that $\dfrac{\partial \ell_i}{\partial \mathrm{f}_i}=\mathrm{sig}[\mathrm{f}_i]-y_i$. (Use $\mathrm{sig}'[z]=\mathrm{sig}[z](1-\mathrm{sig}[z])$.) Then implement both functions below. The loss must be **numerically stable**: no `log(0)` for large $|f|$. Hint: $-\log\mathrm{sig}[f]=\mathrm{softplus}[-f]$, and `np.logaddexp(0, z)` computes $\log(1+e^{z})$ safely.
''')

exercise("5.3",
r'''
def bce_from_logits(f, y):
    """Sum of binary cross-entropy terms, computed stably from logits f."""
    # TODO
    raise NotImplementedError

def bce_grad_logits(f, y):
    """d(loss)/df, elementwise."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t53():
    f = np.array([-3.0, 0.2, 4.0, 40.0, -40.0]); y = np.array([0, 1, 1, 0, 1.0])
    sg = 1 / (1 + np.exp(-f[:3]))
    ref3 = np.sum(-(1 - y[:3]) * np.log(1 - sg) - y[:3] * np.log(sg))
    stable = np.isfinite(bce_from_logits(f, y)) and np.isclose(bce_from_logits(f[3:], y[3:]), 80, atol=1e-6)
    g_num = num_grad(lambda q: bce_from_logits(q, y), f)
    return np.isclose(bce_from_logits(f[:3], y[:3]), ref3) and stable and close(bce_grad_logits(f, y), g_num, 1e-5)
check("5.3", 2, _t53, "sig(f) - y: the error signal is just 'prediction minus truth'.",
      "loss = sum(y*logaddexp(0,-f) + (1-y)*logaddexp(0,f)); grad = 1/(1+exp(-f)) - y")
''',
r'''
def bce_from_logits(f, y):
    return np.sum(y * np.logaddexp(0, -f) + (1 - y) * np.logaddexp(0, f))
def bce_grad_logits(f, y):
    return 1 / (1 + np.exp(-f)) - y
''')

code(r'''
# 💻 Plot the loss vs predicted probability (book P5.2), then fit logistic regression from scratch
lam = np.linspace(1e-3, 1 - 1e-3, 300)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(lam, -np.log(1 - lam), label="y = 0", color=TB["blue"]); ax[0].plot(lam, -np.log(lam), label="y = 1", color=TB["orange"])
ax[0].set_xlabel(r"$\mathrm{sig}[f]$"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Confidently wrong = very expensive")

rr = np.random.default_rng(7); food = rr.uniform(0, 30, 120)
guard = (rr.uniform(size=120) < 1 / (1 + np.exp(-(food - 15) / 3))).astype(float)
phi = np.zeros(2)
for _ in range(5000):
    fz = phi[0] + phi[1] * (food - 15) / 10
    gz = bce_grad_logits(fz, guard)
    phi -= 0.01 * np.array([gz.sum(), (gz * (food - 15) / 10).sum()])
xs = np.linspace(0, 30, 200)
ax[1].scatter(food, guard + rr.normal(0, 0.02, 120), s=10, color=TB["grey"], label="observed")
ax[1].plot(xs, 1 / (1 + np.exp(-(phi[0] + phi[1] * (xs - 15) / 10))), color=TB["green"], lw=2, label="Pr(guard | food)")
ax[1].set_xlabel("food eaten in last hour (g)"); ax[1].legend(); ax[1].set_title("Selfish sentinels"); plt.tight_layout(); plt.show()
''')

md(r'''
## 5C · Multiclass classification: 🦅🐺🐍 decoding alarm calls
Meerkats give **different alarm calls for different predators**. Two acoustic features (peak frequency and call duration) should let us classify the call as **eagle / jackal / cobra**.

- **Categorical** distribution with $Pr(y=k|\mathbf{x})=\mathrm{softmax}_k[\mathbf{f}[\mathbf{x},\boldsymbol\phi]]=\dfrac{\exp[\mathrm{f}_k]}{\sum_{k'}\exp[\mathrm{f}_{k'}]}$ (eq 5.22).
- **Multiclass cross-entropy** (eq 5.24): $L=-\sum_i\Big(\mathrm{f}_{y_i}[\mathbf{x}_i,\boldsymbol\phi]-\log\sum_k\exp\big[\mathrm{f}_k[\mathbf{x}_i,\boldsymbol\phi]\big]\Big)$.
- Its gradient with respect to the logits is $\mathrm{softmax}[\mathbf{f}]-\mathbf{e}_{y}$, where $\mathbf{e}_y$ is the one-hot label. It's the same "prediction minus truth" as in 5B.

### 🧩 Puzzle 5.4 ⭐ A softmax that never overflows
`np.exp(1000)` is `inf`. Use the fact that $\mathrm{softmax}[\mathbf{z}]=\mathrm{softmax}[\mathbf{z}-c]$ for any constant $c$. Columns are examples, so normalise over **axis 0**.
''')

exercise("5.4",
r'''
def softmax(z):
    """z: (K, N) logits -> (K, N) probabilities, normalised over axis 0, stable for huge z."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t54():
    z = np.array([[1000., 1.], [1001., 2.], [1002., 3.]])
    p = softmax(z); ref = np.exp(z[:, 1]) / np.exp(z[:, 1]).sum()
    return np.all(np.isfinite(p)) and close(p[:, 0], ref) and close(p[:, 1], ref) and close(p.sum(0), [1, 1])
check("5.4", 1, _t54, "Subtracting the max: the oldest trick in numerical deep learning.",
      "z = z - z.max(axis=0, keepdims=True)")
''',
r'''
def softmax(z):
    e = np.exp(z - z.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)
''')

code(r'''
# 💻 Linear softmax classifier from scratch: y = softmax(beta + Omega x), trained by gradient descent
rr = np.random.default_rng(11)
centres = np.array([[4.2, 0.20], [2.0, 0.55], [3.1, 1.10]])      # (kHz, s)  eagle / jackal / cobra
names = ["eagle", "jackal", "cobra"]; cols = [TB["blue"], TB["orange"], TB["green"]]
Xc = np.concatenate([rr.normal(c, [0.45, 0.15], (80, 2)) for c in centres]).T          # (2, 240)
yc = np.repeat([0, 1, 2], 80); mu_, sd_ = Xc.mean(1, keepdims=True), Xc.std(1, keepdims=True)
Xn = (Xc - mu_) / sd_; onehot = np.eye(3)[yc].T
Om, be = np.zeros((3, 2)), np.zeros((3, 1)); losses = []
for it in range(600):
    P = softmax(be + Om @ Xn); G = P - onehot                         # dL/df  (3, N)
    Om -= 0.5 * G @ Xn.T / Xn.shape[1]; be -= 0.5 * G.mean(1, keepdims=True)
    losses.append(-np.mean(np.log(P[yc, np.arange(len(yc))])))
g1, g2 = np.meshgrid(np.linspace(0.5, 5.8, 300), np.linspace(-0.3, 1.8, 300))
G = (np.stack([g1.ravel(), g2.ravel()]) - mu_) / sd_; pred = softmax(be + Om @ G).argmax(0).reshape(g1.shape)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].contourf(g1, g2, pred, levels=[-.5, .5, 1.5, 2.5], colors=cols, alpha=0.18)
for k in range(3): ax[0].scatter(*Xc[:, yc == k], s=10, color=cols[k], label=names[k])
ax[0].set_xlabel("peak frequency (kHz)"); ax[0].set_ylabel("duration (s)"); ax[0].legend(); ax[0].set_title("Alarm-call decoder")
ax[1].plot(losses, color=TB["green"]); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("mean cross-entropy"); plt.show()
print("training accuracy:", np.mean(softmax(be + Om @ Xn).argmax(0) == yc))
''')

md(r'''
## 5D · Counts: 🦂 scorpions per hour (book P5.6)
Meerkats eat scorpions (they're largely immune to the venom). The number caught per hour, $y\in\{0,1,2,\dots\}$, rises with temperature. Use a **Poisson** distribution $Pr(y=k)=\frac{\lambda^ke^{-\lambda}}{k!}$, with the rate predicted through an $\exp$ so that it stays positive: $\lambda=\exp[\mathrm{f}[x,\boldsymbol\phi]]$.

### 🧩 Puzzle 5.5 ⭐⭐ Derive and fit the Poisson loss
1. Show that $L=\sum_i\big[\exp(\mathrm{f}_i)-y_i\mathrm{f}_i+\log(y_i!)\big]$ and $\partial\ell_i/\partial\mathrm{f}_i=\exp(\mathrm{f}_i)-y_i$ (prediction minus truth again!).
2. Implement both, then fit $\mathrm{f}=\phi_0+\phi_1 x$ by gradient descent, with $x=(T-25)/10$ the scaled temperature. The true values are $\phi=[1.2, 0.6]$; can you recover them?
''')

exercise("5.5",
r'''
def poisson_nll(y, f):
    # TODO  (math.lgamma(k+1) = log(k!))
    raise NotImplementedError

def poisson_grad_f(y, f):
    # TODO
    raise NotImplementedError

def fit_poisson(x, y, lr=1e-3, n_steps=5000):
    """Gradient descent on [phi0, phi1]; return np.array([phi0, phi1])."""
    # TODO
    raise NotImplementedError
''',
r'''
rr = np.random.default_rng(21); temp = rr.uniform(12, 40, 300); xt = (temp - 25) / 10
scorpions = rr.poisson(np.exp(1.2 + 0.6 * xt))
def _t55():
    f = np.array([0.1, 1.0, 2.0]); y = np.array([0, 3, 7])
    ref = np.sum(np.exp(f) - y * f + np.array([math.lgamma(k + 1) for k in y]))
    ok1 = np.isclose(poisson_nll(y, f), ref) and close(poisson_grad_f(y, f), num_grad(lambda q: poisson_nll(y, q), f), 1e-5)
    phi = fit_poisson(xt, scorpions); print("   recovered phi:", np.round(phi, 3))
    return ok1 and np.all(np.abs(phi - [1.2, 0.6]) < 0.08)
check("5.5", 2, _t55, "Canonical link + matching loss always gives (prediction - truth).",
      "grad wrt phi = sum((exp(f)-y) * [1, x])")
''',
r'''
def poisson_nll(y, f):
    return np.sum(np.exp(f) - y * f + np.array([math.lgamma(k + 1) for k in np.atleast_1d(y)]))
def poisson_grad_f(y, f):
    return np.exp(f) - y
def fit_poisson(x, y, lr=1e-3, n_steps=5000):
    phi = np.zeros(2)
    for _ in range(n_steps):
        g = poisson_grad_f(y, phi[0] + phi[1] * x)
        phi -= lr * np.array([g.sum(), (g * x).sum()])
    return phi
''')

md(r'''
## 5E · Directions: 🧭 which way is the burrow? (book P5.3)
At dusk each foraging group heads home. The heading $y\in(-\pi,\pi]$ is an **angle**, and angles wrap around: $+179°$ and $-179°$ are only 2° apart. The **von Mises** distribution handles this:
$$Pr(y|\mu,\kappa)=\frac{\exp[\kappa\cos(y-\mu)]}{2\pi\,\mathrm{Bessel}_0[\kappa]}\quad\Rightarrow\quad L=-\sum_i\kappa\cos(y_i-\mu_i)+\text{const}.$$

### 🧩 Puzzle 5.6 ⭐⭐⭐ The naive average sends the mob the wrong way
For a constant model $\mu$, set $\partial L/\partial\mu=0$ and show that $\sum_i\sin(y_i-\mu)=0$, which gives $\hat\mu=\mathrm{atan2}\big(\sum_i\sin y_i,\ \sum_i\cos y_i\big)$. Implement it, and compare with `np.mean` on headings that cluster around due **west** ($\pm\pi$).
''')

exercise("5.6",
r'''
def vonmises_mu_mle(angles):
    # TODO
    raise NotImplementedError
''',
r'''
def _t56():
    rr = np.random.default_rng(3)
    heads = np.angle(np.exp(1j * (np.pi + rr.normal(0, 0.3, 50))))      # around ±pi, wrapped
    mu = vonmises_mu_mle(heads); naive = heads.mean()
    print(f"   naive mean = {np.degrees(naive):7.1f}°   von Mises MLE = {np.degrees(mu):7.1f}°   (truth: ±180°)")
    grid = np.linspace(-np.pi, np.pi, 100001); Lg = [-np.sum(np.cos(heads - m)) for m in grid[::100]]
    m_best = grid[::100][np.argmin(Lg)]
    return abs(np.angle(np.exp(1j * (mu - m_best)))) < 0.01
check("5.6", 3, _t56, "Least squares on angles would send the mob off due EAST.",
      "np.arctan2(np.sin(a).sum(), np.cos(a).sum())")
''',
r'''
def vonmises_mu_mle(angles):
    return np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())
''')

md(r'''
### 📖 Cross-entropy, KL divergence and why it's all the same thing (§5.7)
Let $q(y)$ be the empirical distribution of the data, a set of spikes at the $y_i$. Minimizing the KL divergence $D_{KL}[q\,\|\,Pr(y|\boldsymbol\theta)]=\int q\log q\,dy-\int q\log Pr(y|\boldsymbol\theta)\,dy$ over $\boldsymbol\theta$ drops the first term, since it doesn't depend on $\boldsymbol\theta$. What's left is the **cross-entropy** $-\frac1I\sum_i\log Pr(y_i|\boldsymbol\theta)$: our negative log-likelihood. So "cross-entropy loss" and "maximum likelihood" are two names for the same objective.
''')
