from nbh import md, code, exercise

# =====================================================================
# PART 1
# =====================================================================
md(r'''
---
# Part 1 · Introduction (Chapter 1)
### 📖 Recap
The book sorts machine learning into three families:

**Supervised learning** learns a mapping from an input $\mathbf{x}$ to an output $\mathbf{y}$ using example pairs $\{\mathbf{x}_i,\mathbf{y}_i\}$. The model is a *family* of equations $\mathbf{y}=\mathbf{f}[\mathbf{x},\boldsymbol\phi]$, and the parameters $\boldsymbol\phi$ pick one member of the family.
- **Regression**: continuous outputs (univariate or multivariate).
- **Classification**: discrete outputs (binary or multiclass).
- Inputs can be tabular, or *structured* (audio, images, text, molecules). Structured inputs are high-dimensional and their elements are related to one another.

**Unsupervised learning** learns from $\{\mathbf{x}_i\}$ alone, with no labels.
- **Generative models** learn to *synthesize* new examples. They can be probabilistic (giving $Pr(\mathbf{x})$) or not.
- **Latent variables** $\mathbf{z}$ capture the underlying structure, so that $\mathbf{x}$ can be generated from a simpler $\mathbf{z}$.

**Reinforcement learning**: an agent takes **actions** in an environment, changes its **state**, and receives **rewards**. The hard part is *temporal credit assignment*: working out which of many past actions earned a delayed reward.

Chapter 1 ends with **ethics**. Bias and fairness, explainability, weaponization, concentration of power and environmental cost all matter at TBio too. What data we train on shapes what the model "believes".
''')

md(r'''
### 🧩 Puzzle 1.1 ⭐ Sort the field-station requests
The station manager emails you eight requests. Label each one with **one** of:
`"regression"`, `"multivariate regression"`, `"binary classification"`, `"multiclass classification"`, `"unsupervised"`, `"generative"`, `"reinforcement"`.
''')

exercise("1.1",
r'''
requests = {
 "A": "Predict a pup's adult body mass (g) from its mass at 30 days",
 "B": "Decide whether a 2-second audio clip contains an alarm call (yes/no)",
 "C": "Identify which predator (eagle, jackal, cobra) an alarm call refers to",
 "D": "Group 500 unlabelled GPS foraging tracks into 'foraging styles'",
 "E": "Synthesize realistic new meerkat calls for a playback experiment",
 "F": "Train a robot sentinel that learns when to stand guard to maximise mob survival",
 "G": "Predict the mob's (latitude, longitude) at sunset",
 "H": "TBio cameo: predict whether a protein forms condensates at a given concentration",
}
my_answers = {
    "A": None, "B": None, "C": None, "D": None,   # TODO: replace None with a label string
    "E": None, "F": None, "G": None, "H": None,
}
''',
r'''
def _t11():
    if any(v is None for v in my_answers.values()): raise NotImplementedError
    key = _K11
    n = sum(my_answers[k].strip().lower() == v for k, v in key.items())
    print(f"   {n}/8 correct")
    return n >= 7
_K11 = dict(A="regression", B="binary classification", C="multiclass classification",
            D="unsupervised", E="generative", F="reinforcement",
            G="multivariate regression", H="binary classification")
check("1.1", 1, _t11, "You know your way around the field station.",
      "Is the output a number, several numbers, a yes/no, one of many classes... or is there no label at all?")
''',
r'''
my_answers = dict(A="regression", B="binary classification", C="multiclass classification",
                  D="unsupervised", E="generative", F="reinforcement",
                  G="multivariate regression", H="binary classification")
print(my_answers)
''')

md(r'''
### 🧩 Puzzle 1.2 ⭐ Why structured inputs need structured models
A collar microphone records **10 seconds** of audio at **16 kHz**. We flatten that into a vector $\mathbf{x}$ and feed it into a *single* fully connected layer with **1,000** hidden units, each with its own bias.
How many parameters is that? Chapter 4 (§4.5.4) uses this kind of number to motivate convolutional networks.
''')

exercise("1.2",
r'''
n_inputs = None        # TODO
n_params_layer = None  # TODO  (weights + biases)
''',
r'''
def _t12():
    if n_inputs is None or n_params_layer is None: raise NotImplementedError
    return n_inputs == 160_000 and n_params_layer == 160_000 * 1000 + 1000
check("1.2", 1, _t12, "160 million parameters for ONE layer. This is why chapter 10 (convolutions) exists.",
      "Inputs = seconds × samples/second. Each hidden unit has one weight per input plus one bias.")
''',
r'''
n_inputs = 10 * 16_000
n_params_layer = n_inputs * 1000 + 1000
print(n_inputs, f"{n_params_layer:,}")
''')

# =====================================================================
# PART 2
# =====================================================================
md(r'''
---
# Part 2 · Supervised learning (Chapter 2): 🍼 the Great Pup Weigh-in
### 📖 Recap
Every morning the station weighs the pups on a kitchen scale (with a boiled-egg bribe). We want to predict **mass gained (g)** from **hours spent foraging** the day before.

**Model** (1D linear regression, eq 2.4):
$$y = \mathrm{f}[x,\boldsymbol\phi] = \phi_0 + \phi_1 x .$$

**Loss**: the least squares loss (eq 2.5) measures the mismatch over the training set:
$$L[\boldsymbol\phi] = \sum_{i=1}^{I}\big(\mathrm{f}[x_i,\boldsymbol\phi]-y_i\big)^2=\sum_{i=1}^{I}\big(\phi_0+\phi_1x_i-y_i\big)^2 .$$

**Training** means finding $\hat{\boldsymbol\phi}=\underset{\boldsymbol\phi}{\mathrm{argmin}}\, L[\boldsymbol\phi]$. **Testing** means measuring performance on *new* data to see how well the model **generalizes** (underfitting / overfitting; much more on this in Part 8).
''')

code(r'''
# 💻 The weigh-in data (synthetic, but the pups are very real in spirit)
rng = np.random.default_rng(2)
I = 20
hours = np.sort(rng.uniform(0.5, 7.5, I))                 # x: foraging hours
gain = 1.5 + 2.2 * hours + rng.normal(0, 2.0, I)          # y: grams gained (true phi = [1.5, 2.2])

def f_linear(x, phi0, phi1):
    return phi0 + phi1 * x

def loss_ls(phi0, phi1, x, y):
    return np.sum((f_linear(x, phi0, phi1) - y) ** 2)

plt.scatter(hours, gain, color=TB["orange"], zorder=3, label="pups")
xx = np.linspace(0, 8, 100)
plt.plot(xx, f_linear(xx, 0, 3), "--", color=TB["grey"], label=r"guess $\phi=[0,3]$")
plt.xlabel("foraging hours $x$"); plt.ylabel("mass gain $y$ (g)"); plt.legend(); plt.title("The Great Pup Weigh-in")
plt.show()
print("Loss of the guess:", round(loss_ls(0, 3, hours, gain), 2))
''')

code(r'''
# 💻 The loss is a SURFACE over parameter space (cf. Fig 2.3)
P0, P1 = np.meshgrid(np.linspace(-6, 10, 201), np.linspace(0, 4.5, 201))
Lgrid = ((P0[..., None] + P1[..., None] * hours - gain) ** 2).sum(-1)
fig, ax = plt.subplots(figsize=(6, 4.5))
cs = ax.contourf(P0, P1, np.log(Lgrid), 30, cmap="viridis")
ax.contour(P0, P1, np.log(Lgrid), 15, colors="w", linewidths=0.4)
plt.colorbar(cs, label="log loss"); ax.set_xlabel(r"intercept $\phi_0$"); ax.set_ylabel(r"slope $\phi_1$")
ax.set_title("Loss landscape: one valley, one minimum"); plt.show()
''')

md(r'''
### 🧩 Puzzle 2.1 ⭐ Which way is downhill? (book P2.1)
Differentiate the loss. Write $r_i=\phi_0+\phi_1x_i-y_i$ for the residual of point $i$. Then
$$\frac{\partial L}{\partial \phi_0}=\;?\qquad\frac{\partial L}{\partial \phi_1}=\;?$$
Work it out on paper, then implement `grad_ls`. The checker compares your answer with a finite-difference estimate.
''')

exercise("2.1",
r'''
def grad_ls(phi0, phi1, x, y):
    """Return (dL/dphi0, dL/dphi1) for the least squares loss."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t21():
    for p in [(0., 3.), (1.5, 2.2), (-4, 0.7)]:
        g = grad_ls(*p, hours, gain)
        gn = num_grad(lambda q: loss_ls(q[0], q[1], hours, gain), p)
        if not close(g, gn, 1e-3): return False
    return True
check("2.1", 1, _t21, "The gradient of a sum is the sum of the gradients.",
      "Chain rule: d(r^2)/dphi = 2 r * dr/dphi, where dr/dphi0 = 1 and dr/dphi1 = x_i.")
''',
r'''
def grad_ls(phi0, phi1, x, y):
    r = phi0 + phi1 * x - y
    return 2 * np.sum(r), 2 * np.sum(r * x)
''')

md(r'''
### 🧩 Puzzle 2.2 ⭐⭐ Closed form (book P2.2)
Set both derivatives to zero:
$$\sum_i(\phi_0+\phi_1x_i-y_i)=0,\qquad \sum_i(\phi_0+\phi_1x_i-y_i)\,x_i=0 .$$
The first gives $\phi_0=\bar y-\phi_1\bar x$. Substitute it into the second and solve for $\phi_1$. You should find a ratio of a *covariance* to a *variance*. Implement `closed_form`.
''')

exercise("2.2",
r'''
def closed_form(x, y):
    """Return (phi0_hat, phi1_hat) minimising the least squares loss."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t22():
    p = closed_form(hours, gain)
    return close(p, np.polyfit(hours, gain, 1)[::-1], 1e-8)
check("2.2", 2, _t22, "Linear regression is convex with a closed form, but neural nets won't be this kind (chapter 6).",
      "phi1 = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)")
''',
r'''
def closed_form(x, y):
    xb, yb = x.mean(), y.mean()
    phi1 = np.sum((x - xb) * (y - yb)) / np.sum((x - xb) ** 2)
    return yb - phi1 * xb, phi1
print(closed_form(hours, gain))
''')

code(r'''
# 💻 Gradient descent: "walk downhill" (Fig 2.4). Watch it zig-zag along a narrow valley.
def gradient_descent_lr(phi, x, y, lr=1e-3, n_steps=4000):
    path = [np.array(phi, float)]
    for _ in range(n_steps):
        g = np.array(grad_ls(*path[-1], x, y))
        path.append(path[-1] - lr * g)
    return np.array(path)

path = gradient_descent_lr([-5, 4.0], hours, gain)
best = closed_form(hours, gain)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].contourf(P0, P1, np.log(Lgrid), 30, cmap="viridis")
ax[0].plot(path[:, 0], path[:, 1], ".-", color=TB["orange"], ms=2, lw=1)
ax[0].plot(*best, "*", color="w", ms=14); ax[0].set_xlabel(r"$\phi_0$"); ax[0].set_ylabel(r"$\phi_1$")
ax[0].set_title("GD path in parameter space")
ax[1].scatter(hours, gain, color=TB["orange"], zorder=3)
for k, c in zip([0, 5, 50, 500, 4000], plt.cm.viridis(np.linspace(0, 0.9, 5))):
    ax[1].plot(xx, f_linear(xx, *path[k]), color=c, label=f"step {k}")
ax[1].legend(); ax[1].set_title("...and the corresponding lines"); plt.show()
print("GD after 4000 steps:", path[-1], " closed form:", np.array(best))
''')

md(r'''
### 🧩 Puzzle 2.3 ⭐⭐⭐ Discriminative vs generative (book P2.3)
The station's statistician insists on a **generative** model: hours are "caused" by gain, so $x=\mathrm{g}[y,\boldsymbol\phi']=\phi'_0+\phi'_1y$, fitted by least squares on $x$. Inference then uses the **inverse**, $y=\mathrm{g}^{-1}[x,\boldsymbol\phi']=(x-\phi'_0)/\phi'_1$.

1. Fit $\boldsymbol\phi'$ (you can reuse `closed_form` with the arguments swapped).
2. Write the implied prediction line as $y = a + b x$ and return `(a, b)`.
3. Is it the same line as the discriminative fit? Look at the plot and work out why. (Hint: which errors does each loss penalise: vertical or horizontal?)
''')

exercise("2.3",
r'''
def generative_line(x, y):
    """Fit x = p0 + p1*y by least squares, then return (a, b) of the inverse line y = a + b*x."""
    # TODO
    raise NotImplementedError
''',
r'''
def _t23():
    a, b = generative_line(hours, gain)
    p1 = np.polyfit(gain, hours, 1)
    return close((a, b), (-p1[1] / p1[0], 1 / p1[0]), 1e-8)
check("2.3", 3, _t23, "Different loss = different answer, even with the same family of lines.",
      "If x = p0 + p1*y, then y = -p0/p1 + (1/p1)*x.")
''',
r'''
def generative_line(x, y):
    p0, p1 = closed_form(y, x)   # regress x on y
    return -p0 / p1, 1 / p1
a, b = generative_line(hours, gain); d0, d1 = closed_form(hours, gain)
plt.scatter(hours, gain, color=TB["orange"], zorder=3)
plt.plot(xx, d0 + d1 * xx, color=TB["green"], label=f"discriminative  y={d0:.2f}+{d1:.2f}x")
plt.plot(xx, a + b * xx, color=TB["purple"], label=f"generative (inverted)  y={a:.2f}+{b:.2f}x")
plt.legend(); plt.title("Same data, two different lines"); plt.show()
print("The generative fit minimises HORIZONTAL errors, so its inverted line is steeper.\n"
      "The two only agree when the points are perfectly collinear (correlation = ±1).")
''')

