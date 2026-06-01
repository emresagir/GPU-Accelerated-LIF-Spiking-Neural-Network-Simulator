#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Match these with your simulation
N = 1024
T = 3000

# File paths
PY_U = "../torch/u_trace.npy"
PY_G = "../torch/g_trace.npy"
PY_S = "../torch/s_trace.npy"

C_U = "../src/u_c.bin"
C_G = "../src/g_c.bin"
C_S = "../src/s_c.bin"

def load_npy(path):
    return np.load(path)

def load_bin(path, dtype, shape):
    arr = np.fromfile(path, dtype=dtype)
    if arr.size != np.prod(shape):
        raise ValueError(f"{path}: expected {np.prod(shape)} elements, got {arr.size}")
    return arr.reshape(shape)

def compare(name, a, b):
    diff = a - b
    max_abs = np.max(np.abs(diff))
    rmse = np.sqrt(np.mean(diff * diff))
    print(f"{name}:")
    print(f"  max |diff| = {max_abs:.8e}")
    print(f"  rmse       = {rmse:.8e}")
    print(f"  allclose    = {np.allclose(a, b, rtol=1e-5, atol=1e-6)}")
    return diff

def main():
    u_py = load_npy(PY_U)          # shape: (T, N)
    g_py = load_npy(PY_G)          # shape: (T, N)
    s_py = load_npy(PY_S)          # shape: (T, N)

    u_c = load_bin(C_U, np.float32, (T, N))
    g_c = load_bin(C_G, np.float32, (T, N))
    s_c = load_bin(C_S, np.uint8,   (T, N))

    print("Shapes:")
    print("  u_py:", u_py.shape, "u_c:", u_c.shape)
    print("  g_py:", g_py.shape, "g_c:", g_c.shape)
    print("  s_py:", s_py.shape, "s_c:", s_c.shape)
    print()

    du = compare("Membrane potential u", u_py, u_c)
    dg = compare("Synaptic current g", g_py, g_c)

    spike_match = (s_py == s_c).mean()
    print("Spikes:")
    print(f"  exact match rate = {spike_match * 100:.2f}%")
    print(f"  all spikes equal = {np.array_equal(s_py, s_c)}")

    t = np.arange(T)

    # Plot a few neurons to keep it readable
    n_show = min(N, 3)
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, u_py[:, i], label="Python u", linewidth=2)
        ax.plot(t, u_c[:, i], "--", label="C u", linewidth=1.5)
        ax.set_ylabel(f"neuron {i}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time step")
    fig.suptitle("Membrane potential comparison: Python vs C", y=1.02)
    fig.tight_layout()

    # Plot this if the timesteps are small like 20.
    # Plot absolute error for all neurons
    # plt.figure(figsize=(12, 4))
    # for i in range(N):
    #     plt.plot(t, np.abs(du[:, i]), label=f"n{i}")
    # plt.xlabel("time step")
    # plt.ylabel("|u_py - u_c|")
    # plt.title("Absolute membrane potential error")
    # plt.grid(True, alpha=0.3)
    # plt.legend(ncol=4, fontsize=8)
    # plt.tight_layout()

    # Optional: spike raster comparison
    # Show only first X neurons and Y timesteps
    show_T = 100
    show_N = 3

    py_times, py_neurons = np.where(s_py[:show_T, :show_N] > 0)
    c_times, c_neurons = np.where(s_c[:show_T, :show_N] > 0)

    plt.figure(figsize=(12, 4))

    plt.scatter(py_times, py_neurons,
                s=25, marker="|", label="Python")

    plt.scatter(c_times, c_neurons,
                s=25, marker="x", label="C")

    plt.xlim(0, show_T)
    plt.ylim(-1, show_N)
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.title(f"First {show_N} neurons, first {show_T} timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()