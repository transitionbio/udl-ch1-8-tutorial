from nbh import md, code, exercise

md(r'''
---
# Part 6 · Fitting models (Chapter 6): 🏃 race to the burrow
### 📖 Recap
**Gradient descent** (eq 6.3): $\boldsymbol\phi\leftarrow\boldsymbol\phi-\alpha\,\dfrac{\partial L}{\partial\boldsymbol\phi}$, where $\alpha$ is the learning rate.

Linear-regression losses are **convex**: every chord lies above the surface, so there's one minimum. Neural network losses are **non-convex**, with local minima and saddle points. The book explores this with the 2-parameter **Gabor model** (eq 6.8):
$$\mathrm{f}[x,\boldsymbol\phi]=\sin[\phi_0+0.06\,\phi_1x]\cdot\exp\!\Big(-\frac{(\phi_0+0.06\,\phi_1x)^2}{32}\Big).$$

**SGD** (eq 6.10) takes each step using a random **minibatch** $\mathcal{B}_t$: $\;\boldsymbol\phi_{t+1}\leftarrow\boldsymbol\phi_t-\alpha\sum_{i\in\mathcal{B}_t}\partial\ell_i/\partial\boldsymbol\phi$. One pass through the whole dataset is an **epoch**.

**Momentum** (eq 6.11): $\;\mathbf{m}_{t+1}\leftarrow\beta\mathbf{m}_t+(1-\beta)\sum_{i\in\mathcal{B}_t}\partial\ell_i/\partial\boldsymbol\phi,\quad \boldsymbol\phi_{t+1}\leftarrow\boldsymbol\phi_t-\alpha\,\mathbf{m}_{t+1}$.
**Nesterov** (eq 6.12) evaluates the gradient at the look-ahead point $\boldsymbol\phi_t-\alpha\beta\,\mathbf{m}_t$ instead.

**Adam** (eqs 6.15–6.17):
$$\mathbf{m}_{t+1}\leftarrow\beta\mathbf{m}_t+(1-\beta)\mathbf{g}_t,\quad \mathbf{v}_{t+1}\leftarrow\gamma\mathbf{v}_t+(1-\gamma)\mathbf{g}_t^2,$$
$$\tilde{\mathbf{m}}_{t+1}=\frac{\mathbf{m}_{t+1}}{1-\beta^{t+1}},\quad\tilde{\mathbf{v}}_{t+1}=\frac{\mathbf{v}_{t+1}}{1-\gamma^{t+1}},\quad \boldsymbol\phi_{t+1}\leftarrow\boldsymbol\phi_t-\alpha\frac{\tilde{\mathbf{m}}_{t+1}}{\sqrt{\tilde{\mathbf{v}}_{t+1}}+\epsilon}.$$
The bias correction fixes the too-small early estimates (since $\mathbf{m}_0=\mathbf{v}_0=\mathbf{0}$). Dividing by $\sqrt{\tilde{\mathbf v}}$ normalises the step size in every direction.
''')

code(r'''
# 💻 The Gabor model and its dataset (28 points, true phi = [0, 16.6], as in Fig 6.3)
def gabor(x, phi):
    u = phi[0] + 0.06 * phi[1] * x
    return np.sin(u) * np.exp(-u ** 2 / 32.0)

rr = np.random.default_rng(8)
xg = np.sort(rr.uniform(-15, 15, 28)); yg = gabor(xg, [0.0, 16.6]) + rr.normal(0, 0.1, 28)
def gabor_loss(phi, x=xg, y=yg):
    return np.sum((gabor(x, phi) - y) ** 2)

A0, A1 = np.meshgrid(np.linspace(-10, 10, 241), np.linspace(2.5, 22.5, 241))
LG = np.array([[gabor_loss([a, b]) for a, b in zip(r0, r1)] for r0, r1 in zip(A0, A1)])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(xg, yg, color=TB["orange"]); xx = np.linspace(-15, 15, 400)
ax[0].plot(xx, gabor(xx, [0, 16.6]), color=TB["green"]); ax[0].set_title("Gabor data")
c = ax[1].contourf(A0, A1, LG, 30, cmap="viridis"); plt.colorbar(c, ax=ax[1])
ax[1].set_xlabel(r"$\phi_0$"); ax[1].set_ylabel(r"$\phi_1$"); ax[1].set_title("Non-convex loss: many valleys"); plt.show()
''')

md(r'''
### 🧩 Puzzle 6.1 ⭐⭐ Gradient of the Gabor loss (book P6.3)
Let $u=\phi_0+0.06\phi_1x$ and $g(u)=\sin u\cdot e^{-u^2/32}$. Then
$$g'(u)=\cos u\; e^{-u^2/32}-\frac{u}{16}\sin u\; e^{-u^2/32},\qquad \frac{\partial \mathrm{f}}{\partial\phi_0}=g'(u),\qquad \frac{\partial \mathrm{f}}{\partial\phi_1}=0.06\,x\,g'(u).$$
Use these and the chain rule to implement `gabor_grad`.
''')

exercise("6.1",
r'''
def gabor_grad(phi, x=xg, y=yg):
    """Return np.array([dL/dphi0, dL/dphi1]) for the least squares Gabor loss."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t61():
    return all(close(gabor_grad(np.array(p)), num_grad(gabor_loss, p), 1e-4) for p in [(0.3, 15.), (-4, 8.), (6, 20.)])
check("6.1", 2, _t61, "That's analytic gradients. Part 7 automates this for any network.",
      "dL/dphi = sum 2*(f - y) * df/dphi")
''',
r'''
def gabor_grad(phi, x=xg, y=yg):
    u = phi[0] + 0.06 * phi[1] * x; e = np.exp(-u ** 2 / 32.0)
    dg = np.cos(u) * e - (u / 16.0) * np.sin(u) * e
    r = 2 * (np.sin(u) * e - y)
    return np.array([np.sum(r * dg), np.sum(r * dg * 0.06 * x)])
''')

code(r'''
# 💻 Full-batch GD vs SGD from the same starting points (Fig 6.5)
def gd(grad_fn, phi, lr, n):
    path = [np.array(phi, float)]
    for _ in range(n): path.append(path[-1] - lr * grad_fn(path[-1]))
    return np.array(path)

def sgd_gabor(phi, lr, n, batch=7, seed=0):
    r = np.random.default_rng(seed); path = [np.array(phi, float)]; order = r.permutation(28); pos = 0
    for t in range(n):
        lr_t = lr * (0.5 ** (t // 600))                        # learning-rate schedule: halve every 600 steps
        if pos + batch > 28: order, pos = r.permutation(28), 0  # new epoch
        idx = order[pos:pos + batch]; pos += batch
        path.append(path[-1] - lr_t * gabor_grad(path[-1], xg[idx], yg[idx]))
    return np.array(path)

starts = [(-6, 20), (3, 15), (2, 19), (1, 12)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, title in zip(ax, ["Gradient descent", "Stochastic gradient descent (batch 7)"]):
    a.contourf(A0, A1, LG, 30, cmap="viridis"); a.set_title(title); a.set_xlabel(r"$\phi_0$")
for s, col in zip(starts, [TB["orange"], "w", TB["red"], TB["lblue"]]):
    p = gd(gabor_grad, s, 0.01, 2000); ax[0].plot(p[:, 0], p[:, 1], ".-", color=col, ms=2, lw=1)
    q = sgd_gabor(s, 0.2, 2000, seed=3); ax[1].plot(q[:, 0], q[:, 1], ".-", color=col, ms=2, lw=0.8)
    print(f"start {s}: GD ends at loss {gabor_loss(p[-1]):6.3f}   SGD ends at loss {gabor_loss(q[-1]):6.3f}")
for a in ax: a.plot(0, 16.6, "*", color="w", ms=14)
ax[0].set_ylabel(r"$\phi_1$"); plt.show()
esc = sum(gabor_loss(sgd_gabor((3, 15), 0.2, 2000, seed=k)[-1]) < 0.5 for k in range(20))
print(f"From (3, 15) GD always gets stuck. SGD reached the global minimum (loss ≈ 0.44) in {esc}/20 random runs.")
print("Lesson: in a 2-parameter landscape the starting point dominates; SGD *can* escape, but there's no guarantee.")
''')

md(r'''
### 🧩 Puzzle 6.2 ⭐ Quick-fire (book P6.9 & P6.11)
1. We run SGD for **1,000 iterations** on a dataset of **100** examples with **batch size 20**. How many epochs is that?
2. A model has **one million** parameters. How many **entries** does its Hessian have, and how many **terabytes** would it take in float32 (4 bytes each; 1 TB = $10^{12}$ bytes)?
''')

exercise("6.2",
r'''
epochs = None          # TODO
hessian_entries = None # TODO
hessian_TB = None      # TODO
''',
r'''
def _t62():
    if None in (epochs, hessian_entries, hessian_TB): raise NotImplementedError
    return epochs == 200 and hessian_entries == 10 ** 12 and np.isclose(hessian_TB, 4)
check("6.2", 1, _t62, "This is why deep learning uses first-order optimisers: no Hessians allowed.",
      "batches per epoch = 100/20")
''',
r'''
epochs = 1000 // (100 // 20)
hessian_entries = (10 ** 6) ** 2
hessian_TB = hessian_entries * 4 / 1e12
''')

md(r'''
### 🧩 Puzzle 6.3 ⭐⭐ Momentum is a weighted average of past gradients (book P6.10)
Unroll the recursion $\mathbf{m}_{t+1}=\beta\mathbf{m}_t+(1-\beta)\mathbf{g}_t$, starting from $\mathbf{m}_0=\mathbf{0}$. The gradient from $k$ steps ago has weight $w_k=\;?$

Implement `momentum_weight(k, beta)`. The checker confirms it against a simulation and checks that the weights sum to 1 as $t\to\infty$.
''')

exercise("6.3",
r'''
def momentum_weight(k, beta):
    # TODO
    raise NotImplementedError
''',
r'''
def _t63():
    beta, T = 0.9, 60; G = np.eye(T); m = np.zeros(T)
    for t in range(T): m = beta * m + (1 - beta) * G[t]      # m after T steps, as a function of each g_t
    sim = m[::-1]                                            # weight of gradient k steps ago
    return close([momentum_weight(k, beta) for k in range(T)], sim, 1e-12) and np.isclose(sum(momentum_weight(k, beta) for k in range(2000)), 1)
check("6.3", 2, _t63, "An exponential moving average with an effective memory of ~1/(1-beta) steps.",
      "w_k = (1 - beta) * beta**k")
''',
r'''
def momentum_weight(k, beta):
    return (1 - beta) * beta ** k
''')

md(r'''
### 🧩 Puzzle 6.4 ⭐⭐ Implement momentum, Nesterov and Adam from the equations
Each optimiser takes `grad_fn(phi) -> gradient` and returns the **path**: an array of shape `(n_steps+1, n_params)` that includes the start. The checker compares your paths with reference implementations.
''')

exercise("6.4",
r'''
def momentum(grad_fn, phi, alpha, beta=0.9, n_steps=100):
    # TODO
    raise NotImplementedError

def nesterov(grad_fn, phi, alpha, beta=0.9, n_steps=100):
    # TODO  (gradient evaluated at phi - alpha*beta*m)
    raise NotImplementedError

def adam(grad_fn, phi, alpha, beta=0.9, gamma=0.99, eps=1e-8, n_steps=100):
    # TODO  (use t+1 in the bias-correction exponents, t = 0, 1, 2, ...)
    raise NotImplementedError
''',
r'''
def _t64():
    gf = lambda p: np.array([0.1 * p[0], 10 * p[1]]) + np.array([np.cos(p[1]), 0.0])
    p0 = np.array([3.0, 1.0])
    # reference trajectories (computed independently)
    m = np.zeros(2); ph = p0.copy(); R1 = [ph.copy()]
    for _ in range(50): m = .9 * m + .1 * gf(ph); ph = ph - .05 * m; R1.append(ph.copy())
    m = np.zeros(2); ph = p0.copy(); R2 = [ph.copy()]
    for _ in range(50): m = .9 * m + .1 * gf(ph - .05 * .9 * m); ph = ph - .05 * m; R2.append(ph.copy())
    m = np.zeros(2); v = np.zeros(2); ph = p0.copy(); R3 = [ph.copy()]
    for t in range(50):
        g = gf(ph); m = .9 * m + .1 * g; v = .99 * v + .01 * g ** 2
        ph = ph - .05 * (m / (1 - .9 ** (t + 1))) / (np.sqrt(v / (1 - .99 ** (t + 1))) + 1e-8); R3.append(ph.copy())
    ok = [close(momentum(gf, p0, .05, .9, 50), R1, 1e-10), close(nesterov(gf, p0, .05, .9, 50), R2, 1e-10),
          close(adam(gf, p0, .05, .9, .99, 1e-8, 50), R3, 1e-10)]
    print("   momentum / nesterov / adam:", ["✅" if o else "❌" for o in ok])
    return all(ok)
check("6.4", 2, _t64, "Three optimisers in ~15 lines. torch.optim does exactly this.",
      "Path must include the starting point and have n_steps+1 rows.")
''',
r'''
def momentum(grad_fn, phi, alpha, beta=0.9, n_steps=100):
    phi = np.array(phi, float); m = np.zeros_like(phi); path = [phi.copy()]
    for _ in range(n_steps):
        m = beta * m + (1 - beta) * grad_fn(phi); phi = phi - alpha * m; path.append(phi.copy())
    return np.array(path)

def nesterov(grad_fn, phi, alpha, beta=0.9, n_steps=100):
    phi = np.array(phi, float); m = np.zeros_like(phi); path = [phi.copy()]
    for _ in range(n_steps):
        m = beta * m + (1 - beta) * grad_fn(phi - alpha * beta * m); phi = phi - alpha * m; path.append(phi.copy())
    return np.array(path)

def adam(grad_fn, phi, alpha, beta=0.9, gamma=0.99, eps=1e-8, n_steps=100):
    phi = np.array(phi, float); m = np.zeros_like(phi); v = np.zeros_like(phi); path = [phi.copy()]
    for t in range(n_steps):
        g = grad_fn(phi); m = beta * m + (1 - beta) * g; v = gamma * v + (1 - gamma) * g ** 2
        mt = m / (1 - beta ** (t + 1)); vt = v / (1 - gamma ** (t + 1))
        phi = phi - alpha * mt / (np.sqrt(vt) + eps); path.append(phi.copy())
    return np.array(path)
''')

code(r'''
# 💻 Compare them on a long, narrow valley (Fig 6.9): steep in phi1, shallow in phi0
valley = lambda p: 0.5 * (0.05 * p[0] ** 2 + 5 * p[1] ** 2)
valley_grad = lambda p: np.array([0.05 * p[0], 5 * p[1]])
start = [-9.0, 1.5]
runs = {"GD (α=0.38)": gd(valley_grad, start, 0.38, 80), "momentum": momentum(valley_grad, start, 0.38, 0.9, 80),
        "Nesterov": nesterov(valley_grad, start, 0.38, 0.9, 80), "Adam": adam(valley_grad, start, 0.6, 0.9, 0.99, n_steps=80)}
V0, V1 = np.meshgrid(np.linspace(-10, 2, 200), np.linspace(-2, 2, 200))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].contour(V0, V1, valley([V0, V1]), 25, cmap="Greys", linewidths=0.6)
for name, p in runs.items():
    ax[0].plot(p[:, 0], p[:, 1], ".-", ms=3, lw=1, label=name)
    ax[1].semilogy([valley(q) + 1e-12 for q in p], label=name)
ax[0].legend(); ax[0].set_title("Paths"); ax[1].set_title("Loss vs step"); ax[1].legend(); plt.show()
''')

md(r'''
### 🏆 Challenge 6 ⭐⭐ Race to the burrow
The **dune function** $L(a,b)=(1-a)^2+20\,(b-a^2)^2$ (a Rosenbrock cousin) has a curved, banana-shaped valley. The burrow is at $(1,1)$ and you start at $(-1.5,\,2)$.

Pick an optimiser and its hyperparameters. You win when the loss drops below $10^{-4}$, and your score is the **number of steps** it took (maximum 3,000). Fewer steps is better.
- 🥉 < 2000 steps
- 🥈 < 600 steps
- 🥇 < 200 steps
''')

code(r'''
dune = lambda p: (1 - p[0]) ** 2 + 20 * (p[1] - p[0] ** 2) ** 2
dune_grad = lambda p: np.array([-2 * (1 - p[0]) - 80 * p[0] * (p[1] - p[0] ** 2), 40 * (p[1] - p[0] ** 2)])

def race(path):
    L = np.array([dune(p) for p in path]); hit = np.flatnonzero(L < 1e-4)
    steps = int(hit[0]) if len(hit) else 10 ** 9
    D0, D1 = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1, 3, 300))
    plt.figure(figsize=(5, 4)); plt.contour(D0, D1, np.log(dune([D0, D1]) + 1e-3), 30, cmap="YlOrBr")
    plt.plot(path[:, 0], path[:, 1], ".-", color=TB["green"], ms=2, lw=0.8); plt.plot(1, 1, "*", color=TB["red"], ms=15)
    plt.title(f"steps to burrow: {steps if steps < 10**9 else 'DNF'}"); plt.show()
    return steps
''')

exercise("C6",
r'''
my_path = gd(dune_grad, [-1.5, 2.0], lr=0.001, n=3000)   # TODO: swap in momentum / nesterov / adam and tune!
''',
r'''
def _tc6():
    assert len(my_path) <= 3001 and np.allclose(my_path[0], [-1.5, 2.0]), "start at (-1.5, 2) and use <= 3000 steps"
    s = race(np.asarray(my_path)); print(f"   {s if s < 10**9 else 'did not finish'} steps → {medal(s, 2000, 600, 200)}")
    return s < 200
check("C6", 2, _tc6, "First into the burrow. The mob is impressed.",
      "Try adam(dune_grad, [-1.5, 2.0], alpha=0.05..0.3, ...) or momentum with beta ≈ 0.9 and a bigger lr.")
''',
r'''
my_path = adam(dune_grad, [-1.5, 2.0], alpha=0.2, beta=0.9, gamma=0.99, n_steps=3000)
''')
