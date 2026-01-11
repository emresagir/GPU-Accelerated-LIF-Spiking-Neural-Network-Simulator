import numpy as np

N = 8          
T = 20

import numpy as np

u_c = np.fromfile("u_c.bin", dtype=np.float32).reshape(T, N)
u_py = np.load("../torch/u_trace.npy")

diff = u_py - u_c
for t in range(T):
    for i in range(N):
        if u_py[t,i] != 0:
            print(f"t={t}, i={i}, u_py={u_py[t,i]:.6f}, u_c={u_c[t,i]:.6f}, diff={diff[t,i]:.6e}")


s_c = np.fromfile("s_c.bin", dtype=np.uint8).reshape(T, N)
s_py = np.load("../torch/s_trace.npy")

for t in range(T):
    for i in range(N):
        if s_py[t,i] == 1:
            print(f"t={t}, i={i}, s_py={s_py[t,i]}, s_c={s_c[t,i]}, match={s_py[t,i] == s_c[t,i]}")