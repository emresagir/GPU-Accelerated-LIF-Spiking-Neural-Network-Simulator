# snn_reference.py
import torch
import numpy as np
import time

# Config (match with C)
# N = 1024           # neurons 
# T = 1000           # timesteps
N = 8           # neurons 
T = 20           # timesteps
dt = 1e-3
tau_m = 20e-3
tau_s = 5e-3
theta = 1.0

device = torch.device("cpu")   

beta  = float(np.exp(-dt / tau_m))
alpha = float(np.exp(-dt / tau_s))

# RNG seed for reproducibility
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# Create weights (post x pre) as in C: W[i][j] is weight from j->i
W = (torch.rand((N, N), dtype=torch.float32, device=device) * 0.05).contiguous()
# Optionally, W sparse or structured
# W = W * (torch.rand((N,N)) < 0.01).float()   # sparsify for testing

# Optional: save weights for C
W.cpu().numpy().astype(np.float32).tofile("W_post_pre.f32")

# states
u = torch.zeros(N, dtype=torch.float32, device=device)
g = torch.zeros(N, dtype=torch.float32, device=device)
s = torch.zeros(N, dtype=torch.uint8, device=device)
s_prev = torch.zeros_like(s)

# External Poisson input parameters
use_poisson_input = True
input_rate_hz = 20.0
p_input = input_rate_hz * dt
ext_weight = 10.2

# Recording 
record_spikes = torch.zeros((T, N), dtype=torch.uint8, device=device)

# External input
ext_spikes_mat = (torch.rand((T, N)) < p_input).to(torch.uint8)   
ext_spikes_mat.numpy().tofile("ext_spikes.u8")

ext_spikes = ext_spikes_mat.to(device)   

# For test
u_trace = torch.zeros((T, N), dtype=torch.float32, device=device)
g_trace = torch.zeros((T, N), dtype=torch.float32, device=device)
s_trace = torch.zeros((T, N), dtype=torch.uint8, device=device)

t0 = time.time()
for t in range(T):

    # recurrent input
    input_current = torch.mv(W, s_prev.float())

    # external input 
    input_current += ext_spikes[t].float() * ext_weight           

    # synapse update: g = alpha * g + input_current
    g.mul_(alpha).add_(input_current)

    # soft reset with previous spikes (match w C)
    u = u - s_prev.float() * theta

    # membrane integrate: u = beta * u + (1-beta) * g
    u = u * beta + (1.0 - beta) * g

    # spike generation
    s = (u > theta).to(torch.uint8)

    # store and advance
    record_spikes[t] = s
    s_prev = s

    # Test
    u_trace[t] = u.clone()
    g_trace[t] = g.clone()
    s_trace[t] = s.clone()

# profiling
torch.cuda.synchronize() if device.type == "cuda" else None
t1 = time.time()
print(f"Completed {T} steps, time = {t1-t0:.4f}s, ms/step = {(t1-t0)/T*1000:.4f}")

# Save final traces or spikes for comparison with C
np.save("spikes_ref.npy", record_spikes.cpu().numpy())
np.save("u_ref.npy", u.cpu().numpy())
print("Saved spikes_ref.npy and u_ref.npy")
print(ext_spikes_mat.float().mean())

# Test 
np.save("u_trace.npy", u_trace.cpu().numpy())
np.save("g_trace.npy", g_trace.cpu().numpy())
np.save("s_trace.npy", s_trace.cpu().numpy())

