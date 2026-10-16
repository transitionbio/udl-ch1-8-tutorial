import nbh
from nbh import md, code
S = nbh.SESSION
if S == 2:
    code(r'''
#@title 📦 Carry-over from Session 1 (run me: re-creates what Parts 5–6 need)
def relu(z):
    return np.maximum(z, 0.0)

# Part 2: the pup weigh-in data and the linear model / least squares loss
rng = np.random.default_rng(2)
I = 20
hours = np.sort(rng.uniform(0.5, 7.5, I))
gain = 1.5 + 2.2 * hours + rng.normal(0, 2.0, I)
def f_linear(x, phi0, phi1):
    return phi0 + phi1 * x
def loss_ls(phi0, phi1, x, y):
    return np.sum((f_linear(x, phi0, phi1) - y) ** 2)
print("Carried over: relu, hours, gain, f_linear, loss_ls 🐾")
''', form=True)
elif S == 3:
    code(r'''
#@title 📦 Carry-over from Sessions 1–2 (run me: re-creates what Parts 7–9 need)
def relu(z):
    return np.maximum(z, 0.0)

# Part 4: the general deep network (eq 4.15) and He-style initialisation
def deep_forward(x, betas, omegas, act=relu):
    all_f, all_h = [], [x]
    K = len(omegas) - 1
    for k in range(K + 1):
        f = betas[k] + omegas[k] @ all_h[-1]
        all_f.append(f)
        if k < K:
            all_h.append(act(f))
    return all_f[-1], all_f, all_h

def init_params(dims, sigma2=None, rng=np.random.default_rng(0)):
    betas, omegas = [], []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        s2 = 2.0 / d_in if sigma2 is None else sigma2
        omegas.append(rng.normal(0, np.sqrt(s2), (d_out, d_in))); betas.append(np.zeros((d_out, 1)))
    return betas, omegas
print("Carried over: relu, deep_forward, init_params 🐾")
''', form=True)
