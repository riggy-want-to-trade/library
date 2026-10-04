import torch, torch.nn as nn
exec(open("_test_generalise.py").read().split("# ---------- PART II . 4")[0])  # machinery + truth setup

# rebuild the exact part-4 setup from the main test
N, M, E = 3, 3, 3
torch.manual_seed(1)
A_t = make_stable_matrix(N)
B_t = torch.randn(N, N_INPUTS) * 0.5
C_t = torch.randn(M, N) * 0.8
d_t = torch.randn(M) * 0.5
Z0 = torch.randn(E, N) * 0.5

def forward(t_, A_, B_, C_, d_, Z0_):
    zs, ys = [], []
    for z0e in Z0_:
        z = solve_ode(rhs_system, z0e, t_, A_, B_)
        zs.append(z)
        ys.append((C_ @ z.T + d_[:, None]).T)
    return torch.stack(zs), torch.stack(ys)

t_obs = torch.linspace(0, 8, 33)
torch.manual_seed(2)
_, Y_obs = forward(t_obs, A_t, B_t, C_t, d_t, Z0)
Y_obs = Y_obs + 0.02 * torch.randn(E, 33, M)

def make_par(seed):
    torch.manual_seed(seed)
    return nn.Parameter(torch.cat([
        make_stable_matrix(N).flatten() * 0.3,
        torch.randn(N, N_INPUTS).flatten() * 0.3,
        torch.randn(M, N).flatten() * 0.3,
        torch.zeros(M)]))

def unpack(p):
    A_ = p[:N*N].reshape(N, N)
    B_ = p[N*N : N*N+N*N_INPUTS].reshape(N, N_INPUTS)
    C_ = p[N*N+N*N_INPUTS : N*N+N*N_INPUTS+M*N].reshape(M, N)
    d_ = p[-M:]
    return A_, B_, C_, d_

def report(tag, p):
    A_f, B_f, C_f, d_f = unpack(p)
    print(f"{tag}: loss-opt |err| A={(A_f-A_t).abs().max().item():.2e} B={(B_f-B_t).abs().max().item():.2e} "
          f"C={(C_f-C_t).abs().max().item():.2e} d={(d_f-d_t).abs().max().item():.2e}")

# (a) Adam longer with lr decay
p = make_par(3)
opt = torch.optim.Adam([p], lr=0.05)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.3)
for it in range(6000):
    A_, B_, C_, d_ = unpack(p)
    _, Y_pred = forward(t_obs, A_, B_, C_, d_, Z0)
    loss = ((Y_pred - Y_obs) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 1500 == 0: print(f"adam  iter {it:4d} loss {loss.item():.3e} lr {sched.get_last_lr()[0]:.1e}")
report("adam-6k", p)

# (b) LBFGS
p = make_par(3)
opt = torch.optim.LBFGS([p], lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
for it in range(200):
    def closure():
        opt.zero_grad()
        A_, B_, C_, d_ = unpack(p)
        _, Y_pred = forward(t_obs, A_, B_, C_, d_, Z0)
        loss = ((Y_pred - Y_obs) ** 2).mean()
        loss.backward()
        return loss
    loss = opt.step(closure)
    if it % 50 == 0: print(f"lbfgs iter {it:4d} loss {loss.item():.3e}")
report("lbfgs", p)
