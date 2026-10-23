from nbh import md, code, exercise

md(r'''
---
# Part 3 · Shallow neural networks (Chapter 3): ⛏️ digging burrows with ReLUs
### 📖 Recap
A shallow network with one input, one output and three hidden units has **ten parameters** (eq 3.1):
$$y=\phi_0+\phi_1\,a[\theta_{10}+\theta_{11}x]+\phi_2\,a[\theta_{20}+\theta_{21}x]+\phi_3\,a[\theta_{30}+\theta_{31}x],$$
with the **ReLU** activation $a[z]=\mathrm{ReLU}[z]=\max(0,z)$ (eq 3.2).

It works in three stages:
1. Compute three **linear functions** of the input; these are the *pre-activations*.
2. **Clip** each one at zero with the ReLU to get the hidden units $h_d$ (the *activations*).
3. Take a **weighted sum** of the hidden units and add an offset $\phi_0$.

The result is a **continuous piecewise-linear** function. Each hidden unit contributes one "joint", at $x=-\theta_{d0}/\theta_{d1}$, so $D$ hidden units give at most $D+1$ linear regions. Each region corresponds to one **activation pattern** (which units are active and which are clipped).

**Universal approximation theorem:** with enough hidden units, a shallow net can approximate any continuous function on a compact domain to arbitrary precision.
''')

code(r'''
# 💻 Build the 10-parameter network. These are the book's own parameters (P3.8 / Fig 3.3).
def relu(z):
    return np.maximum(z, 0.0)

def shallow_1_1_3(x, act, phi, theta):
    """phi = [phi0, phi1, phi2, phi3]; theta = 3x2 array [[th10, th11], [th20, th21], [th30, th31]].
    Returns output y plus the intermediate pre-activations and activations (for plotting)."""
    pre = theta[:, [0]] + theta[:, [1]] * x[None, :]     # (3, N) linear functions
    h = act(pre)                                         # (3, N) hidden units
    y = phi[0] + phi[1:] @ h                             # weighted sum + offset
    return y, pre, h

PHI_BOOK = np.array([-0.23, -1.3, 1.3, 0.66])
THETA_BOOK = np.array([[-0.2, 0.4], [-0.9, 0.9], [1.1, -0.7]])

def plot_decomposition(act, phi, theta, title=""):
    x = np.linspace(0, 2, 400); y, pre, h = shallow_1_1_3(x, act, phi, theta)
    fig, ax = plt.subplots(3, 4, figsize=(12, 6.5), sharex=True)
    for d in range(3):
        ax[d, 0].plot(x, pre[d], color=TB["blue"]); ax[d, 0].set_title(fr"$\theta_{{{d+1}0}}+\theta_{{{d+1}1}}x$")
        ax[d, 1].plot(x, h[d], color=TB["green"]); ax[d, 1].set_title(fr"$h_{d+1}$")
        ax[d, 2].plot(x, phi[d + 1] * h[d], color=TB["orange"]); ax[d, 2].set_title(fr"$\phi_{d+1}h_{d+1}$")
        for c in range(3): ax[d, c].set_ylim(-1, 1)
    for d in range(3): ax[d, 3].axis("off")
    big = fig.add_subplot(1, 4, 4); big.plot(x, y, color="k", lw=2)
    big.set_title(r"$y=\phi_0+\sum_d\phi_dh_d$"); big.set_ylim(-1, 1)
    fig.suptitle(title or "Fig 3.3 recreated", fontweight="bold"); plt.tight_layout(); plt.show()

plot_decomposition(relu, PHI_BOOK, THETA_BOOK)
''')

md(r'''
### 🧩 Puzzle 3.1 ⭐ Where are the joints? What are the slopes? (book P3.3)
- Hidden unit $d$ switches on or off where $\theta_{d0}+\theta_{d1}x=0$, i.e. at $x=-\theta_{d0}/\theta_{d1}$.
- Inside a region, the slope is $\sum_{d\,\in\,\text{active}}\phi_d\,\theta_{d1}$.

Implement both functions. `region_slopes` must return the slopes from **left to right** over the whole real line: four values for three joints.
''')

exercise("3.1",
r'''
def joints(theta):
    """Sorted x-positions of the joints."""
    # TODO
    raise NotImplementedError

def region_slopes(phi, theta):
    """Slopes of the len(theta)+1 linear regions, left to right."""
    # TODO  (tip: pick a test point inside each region and ask which units are active)
    raise NotImplementedError
''',
r'''
def _t31():
    j = joints(THETA_BOOK); s = region_slopes(PHI_BOOK, THETA_BOOK)
    x = np.array([j[0] - 1, (j[0] + j[1]) / 2, (j[1] + j[2]) / 2, j[2] + 1]); e = 1e-5
    num = (shallow_1_1_3(x + e, relu, PHI_BOOK, THETA_BOOK)[0] - shallow_1_1_3(x - e, relu, PHI_BOOK, THETA_BOOK)[0]) / (2 * e)
    print("   joints:", np.round(j, 4), " slopes:", np.round(s, 4))
    return close(j, [0.5, 1.0, 1.1 / 0.7], 1e-6) and close(s, num, 1e-5)
check("3.1", 1, _t31, "You can read a ReLU net like a map.",
      "joints = -theta[:,0]/theta[:,1]; a unit is active where theta0 + theta1*x > 0.")
''',
r'''
def joints(theta):
    return np.sort(-theta[:, 0] / theta[:, 1])

def region_slopes(phi, theta):
    j = joints(theta)
    tests = np.concatenate([[j[0] - 1], (j[:-1] + j[1:]) / 2, [j[-1] + 1]])
    slopes = []
    for x in tests:
        active = theta[:, 0] + theta[:, 1] * x > 0
        slopes.append(np.sum(phi[1:][active] * theta[active, 1]))
    return np.array(slopes)
print(joints(THETA_BOOK), region_slopes(PHI_BOOK, THETA_BOOK))
''')

md(r'''
### 🧩 Puzzle 3.2 ⭐⭐ Slopes aren't independent (book P3.9 + P3.2)
With 3 units there are four regions but only three independent slopes. Show that **slope(region 3) = slope(region 1) + slope(region 4)** for the Fig 3.3 network.

Then record the **activation pattern** of each region (which units are on), as a list of four sets using unit numbers 1–3.

*Why does the identity hold? The three units' "switch-on" directions differ: units 1 and 2 have positive $\theta_{d1}$ and switch on to the right, while unit 3 has negative $\theta_{31}$ and switches on to the left.*
''')

exercise("3.2",
r'''
activation_patterns = [None, None, None, None]   # TODO e.g. [{3}, {...}, {...}, {...}]
''',
r'''
def _t32():
    if any(p is None for p in activation_patterns): raise NotImplementedError
    s = region_slopes(PHI_BOOK, THETA_BOOK)
    ok_id = np.isclose(s[2], s[0] + s[3])
    return ok_id and [set(p) for p in activation_patterns] == [{3}, {1, 3}, {1, 2, 3}, {1, 2}]
check("3.2", 2, _t32, "Region 3 has all three units on; regions 1 and 4 split them between them.",
      "Plug test points into theta0 + theta1*x > 0 for each unit.")
''',
r'''
activation_patterns = [{3}, {1, 3}, {1, 2, 3}, {1, 2}]
s = region_slopes(PHI_BOOK, THETA_BOOK)
print("slopes", s, "| s3 == s1 + s4 ?", np.isclose(s[2], s[0] + s[3]))
# Proof: s1 = phi3*th31 (only unit 3), s4 = phi1*th11 + phi2*th21 (units 1,2), s3 = all three = s1 + s4.
''')

md(r'''
### 🧩 Puzzle 3.3 ⭐⭐ Non-negative homogeneity (book P3.5 & P3.6)
For $\alpha>0$, $\mathrm{ReLU}[\alpha z]=\alpha\,\mathrm{ReLU}[z]$. So if we multiply $\theta_{d0},\theta_{d1}$ by $\alpha$ and divide $\phi_d$ by $\alpha$, the function shouldn't change. Implement `rescale_unit`, then try $\alpha=-1$ and see what breaks.

*Consequence: the loss surface has whole curves of equivalent minima, so the least squares minimum is **not unique** (book P3.7).*
''')

exercise("3.3",
r'''
def rescale_unit(phi, theta, d, alpha):
    """Return new (phi, theta) with hidden unit d (0-based) rescaled by alpha (input side) and 1/alpha (output side)."""
    # TODO (don't modify the inputs in place!)
    raise NotImplementedError
''',
r'''
def _t33():
    x = np.linspace(-3, 3, 301); y0 = shallow_1_1_3(x, relu, PHI_BOOK, THETA_BOOK)[0]
    p2, t2 = rescale_unit(PHI_BOOK, THETA_BOOK, 1, 3.7)
    same = close(shallow_1_1_3(x, relu, p2, t2)[0], y0, 1e-9)
    p3, t3 = rescale_unit(PHI_BOOK, THETA_BOOK, 1, -1.0)
    differs = not close(shallow_1_1_3(x, relu, p3, t3)[0], y0, 1e-3)
    untouched = close(THETA_BOOK, [[-0.2, 0.4], [-0.9, 0.9], [1.1, -0.7]])
    return same and differs and untouched
check("3.3", 2, _t33, "With alpha < 0 the unit flips which side of its joint is active: a different function.",
      "copy first: phi = phi.copy(); theta = theta.copy()")
''',
r'''
def rescale_unit(phi, theta, d, alpha):
    phi, theta = phi.copy(), theta.copy()
    theta[d] *= alpha; phi[d + 1] /= alpha
    return phi, theta
''')

code(r'''
# 💻 The activation-function zoo (Fig 3.13)
from math import erf
zoo = {
    "ReLU": relu,
    "leaky ReLU (0.1)": lambda z: np.where(z > 0, z, 0.1 * z),
    "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
    "tanh": np.tanh,
    "softplus": lambda z: np.log1p(np.exp(z)),
    "GELU": lambda z: 0.5 * z * (1 + np.vectorize(erf)(z / np.sqrt(2))),
    "Swish (β=1)": lambda z: z / (1 + np.exp(-z)),
    "HardSwish": lambda z: np.where(z < -3, 0, np.where(z > 3, z, z * (z + 3) / 6)),
}
z = np.linspace(-4, 4, 400)
fig, axs = plt.subplots(2, 4, figsize=(12, 5), sharex=True)
for ax, (name, fn) in zip(axs.ravel(), zoo.items()):
    ax.plot(z, fn(z), color=TB["green"], lw=2); ax.set_title(name); ax.axhline(0, color="k", lw=0.5)
plt.tight_layout(); plt.show()
''')

md(r'''
### 🧩 Puzzle 3.4 ⭐ Swap the activation (book P3.8)
Re-run `plot_decomposition` with the three activations below, then match each one to the family of functions it produces.

$$\text{heaviside}[z]=\begin{cases}0&z<0\\1&z\ge0\end{cases},\qquad \text{rect}[z]=\begin{cases}0&z<0\\1&0\le z\le1\\0&z>1\end{cases},\qquad\tanh[z].$$

Families:
- **(a)** smooth sums of S-shaped curves
- **(b)** piecewise-constant staircases with up to 3 jumps
- **(c)** sums of up to 3 rectangular "boxes" (up to 6 jumps)
''')

exercise("3.4",
r'''
heaviside = lambda z: (z >= 0).astype(float)
rect = lambda z: ((z >= 0) & (z <= 1)).astype(float)
for nm, fn in [("heaviside", heaviside), ("rect", rect), ("tanh", np.tanh)]:
    plot_decomposition(fn, PHI_BOOK, THETA_BOOK, title=f"activation = {nm}")

family = {"heaviside": None, "rect": None, "tanh": None}   # TODO: "a", "b" or "c"
''',
r'''
def _t34():
    if None in family.values(): raise NotImplementedError
    return family == {"heaviside": "b", "rect": "c", "tanh": "a"}
check("3.4", 1, _t34, "Heaviside nets were the original perceptrons, and you can't train them by gradient descent. Why not?",
      "Look at the right-hand column of each figure.")
''',
r'''
family = {"heaviside": "b", "rect": "c", "tanh": "a"}
''')

md(r'''
### 🧩 Puzzle 3.5 ⭐⭐ Universal approximation, by construction
The **sentinel vigilance curve** $v(t)$ gives how alert the mob's sentinel is through the day ($t\in[0,1]$).

Build a shallow net that **interpolates** $v$ at $D+1$ knots $t_0<\dots<t_D$. Use hidden units $h_d=\mathrm{ReLU}[t-t_d]$ for $d=0,\dots,D-1$ (so $\theta_{d0}=-t_d$, $\theta_{d1}=1$), set $\phi_0=v(t_0)$, and choose $\phi_d$ so that the slope *changes* by the right amount at each knot:
$$\phi_{d}= s_d - s_{d-1},\quad s_d=\frac{v(t_{d+1})-v(t_d)}{t_{d+1}-t_d},\quad s_{-1}=0 .$$
This is the argument behind Fig 3.5: more units, more regions, smaller error.
''')

code(r'''
def vigilance(t):
    return 0.6 + 0.3 * np.sin(2 * np.pi * t) * np.exp(-1.5 * t) + 0.15 * np.cos(9 * t)

def shallow_1d(x, phi0, phis, theta0s, theta1s):
    """General 1-input/1-output shallow ReLU net with D units."""
    return phi0 + relu(theta0s[:, None] + theta1s[:, None] * x[None, :]).T @ phis
''')

exercise("3.5",
r'''
def make_interpolating_net(tk, vk):
    """Knots tk (D+1,) and values vk (D+1,) -> (phi0, phis, theta0s, theta1s) with D hidden units."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t35():
    tk = np.linspace(0, 1, 51); net = make_interpolating_net(tk, vigilance(tk))
    t = np.linspace(0, 1, 2001)
    err = np.max(np.abs(shallow_1d(t, *net) - vigilance(t)))
    knots = close(shallow_1d(tk, *net), vigilance(tk), 1e-9)
    print(f"   max error with 50 units: {err:.4f}")
    return knots and err < 0.01
check("3.5", 2, _t35, "Universal approximation, demonstrated by construction.",
      "slopes s = np.diff(vk)/np.diff(tk); phis = np.diff(np.concatenate([[0], s]))")
''',
r'''
def make_interpolating_net(tk, vk):
    s = np.diff(vk) / np.diff(tk)
    phis = np.diff(np.concatenate([[0.0], s]))
    return vk[0], phis, -tk[:-1], np.ones(len(tk) - 1)
''')

code(r'''
# 💻 More units -> more regions -> smaller error
t = np.linspace(0, 1, 1000)
fig, axs = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
for ax, D in zip(axs, [3, 6, 12, 48]):
    tk = np.linspace(0, 1, D + 1); y = shallow_1d(t, *make_interpolating_net(tk, vigilance(tk)))
    ax.plot(t, vigilance(t), "--", color=TB["grey"]); ax.plot(t, y, color=TB["green"], lw=2)
    ax.set_title(f"D={D}  max err={np.max(np.abs(y - vigilance(t))):.3f}")
axs[0].set_ylabel("vigilance"); plt.tight_layout(); plt.show()
''')

md(r'''
### 📖 Multivariate inputs: regions become convex polygons
With $D_i=2$ inputs, each hidden unit's joint becomes a **line** across the input plane, and the regions become **convex polygons** (Fig 3.8). **Zaslavsky (1975):** $D$ hyperplanes cut $\mathbb{R}^{D_i}$ into at most
$$\sum_{j=0}^{D_i}\binom{D}{j}$$
regions. Below we colour each pixel of the plane by its activation pattern and count the distinct colours.
''')

code(r'''
def count_regions_2d(Om, b, lim=4, n=600):
    g = np.linspace(-lim, lim, n); X1, X2 = np.meshgrid(g, g)
    pre = b[:, None] + Om @ np.stack([X1.ravel(), X2.ravel()])
    codes = (pre > 0).astype(int).T @ (2 ** np.arange(len(b)))
    return codes.reshape(n, n), len(np.unique(codes))

r = np.random.default_rng(5)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, D in zip(axs, [3, 5, 12]):
    img, n = count_regions_2d(r.normal(size=(D, 2)), r.normal(size=D))
    ax.imshow(img % 17, cmap="tab20", origin="lower", extent=[-4, 4, -4, 4])
    ax.set_title(f"D={D} hidden units → {n} regions"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
plt.tight_layout(); plt.show()
''')

md(r'''
### 🧩 Puzzle 3.6 ⭐ Counting regions and parameters (book P3.17 & P3.18)
1. `max_regions(D, Di)`: Zaslavsky's bound. Note that the sum stops at $\min(D_i, D)$.
2. `n_params_shallow(Di, D, Do)`: a general shallow network (eqs 3.11–3.12) has how many weights and biases?

The checker uses the book's numbers: Fig 3.11 has twenty parameters, and $D=500$ units on a $D_i=100$-dimensional input give more than $10^{107}$ regions.
''')

exercise("3.6",
r'''
def max_regions(D, Di):
    # TODO (math.comb is your friend)
    raise NotImplementedError

def n_params_shallow(Di, D, Do):
    # TODO
    raise NotImplementedError
''',
r'''
def _t36():
    return (max_regions(3, 2) == 7 and max_regions(5, 2) == 16 and max_regions(4, 1) == 5
            and max_regions(3, 3) == 8 and max_regions(500, 100) > 10 ** 107
            and n_params_shallow(3, 3, 2) == 20 and n_params_shallow(1, 4, 2) == 18
            and n_params_shallow(2, 3, 1) == 13 and n_params_shallow(100, 500, 1) == 51001)
check("3.6", 1, _t36, "51,001 parameters can carve 10^107 regions. Small network, huge function space.",
      "Each hidden unit has Di weights + 1 bias; each output has D weights + 1 bias.")
''',
r'''
def max_regions(D, Di):
    return sum(math.comb(D, j) for j in range(min(Di, D) + 1))

def n_params_shallow(Di, D, Do):
    return D * (Di + 1) + Do * (D + 1)
''')

md(r'''
### 🏆 Challenge 3 ⭐⭐ Dig the burrow
The mob's main burrow has a known cross-section (the dashed line below). Hand-pick the ten parameters of a **3-unit** shallow net so that it matches the burrow profile on $x\in[0,1]$.
- 🥉 MSE < 0.01
- 🥈 MSE < 0.001
- 🥇 exact (MSE < 1e-10)

*Strategy: find the three joints and the slope changes at each joint, then use units of the form $\phi_d\,\mathrm{ReLU}[x-c_d]$.*
''')

code(r'''
def burrow(x):
    return np.select([x < 0.2, x < 0.5, x < 0.7], [0.0, -2 * (x - 0.2), -0.6 + 3 * (x - 0.5)], 0.0)
xb = np.linspace(0, 1, 501)
plt.figure(figsize=(6, 2.6)); plt.fill_between(xb, burrow(xb), -0.8, color=TB["sand"])
plt.plot(xb, burrow(xb), "k--"); plt.title("Burrow cross-section (sand in beige)"); plt.ylim(-0.8, 0.2); plt.show()
''')

exercise("C3",
r'''
my_phi = np.array([0.0, 0.0, 0.0, 0.0])                  # TODO [phi0, phi1, phi2, phi3]
my_theta = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])  # TODO [[th10, th11], ...]
''',
r'''
def _tc3():
    mse = np.mean((shallow_1_1_3(xb, relu, my_phi, my_theta)[0] - burrow(xb)) ** 2)
    print(f"   MSE = {mse:.2e}  →  {medal(mse, 1e-2, 1e-3, 1e-10)}")
    plt.figure(figsize=(6, 2.6)); plt.plot(xb, burrow(xb), "k--", label="burrow")
    plt.plot(xb, shallow_1_1_3(xb, relu, my_phi, my_theta)[0], color=TB["green"], lw=2, label="your net")
    plt.legend(); plt.show()
    return mse < 1e-10
check("C3", 2, _tc3, "Perfect burrow. The mob moves in tonight.",
      "Joints at 0.2, 0.5, 0.7. Slope goes 0 → -2 → +3 → 0, so the slope changes are -2, +5, -3.")
''',
r'''
my_phi = np.array([0.0, -2.0, 5.0, -3.0])
my_theta = np.array([[-0.2, 1.0], [-0.5, 1.0], [-0.7, 1.0]])
''')
