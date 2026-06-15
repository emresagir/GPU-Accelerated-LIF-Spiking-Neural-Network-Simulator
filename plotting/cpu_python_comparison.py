#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Match these 
N = 8
T = 20

# File paths
PY_U = "../torch/u_trace.npy"
PY_G = "../torch/g_trace.npy"
PY_S = "../torch/s_trace.npy"

C_U = "../src/u_c.bin"
C_G = "../src/g_c.bin"
C_S = "../src/s_c.bin"

GPU_U = "../gpu_version/u_gpu.bin"
GPU_G = "../gpu_version/g_gpu.bin"
GPU_S = "../gpu_version/s_gpu.bin"

# ----------------------------
# Plot controls
# ----------------------------
show_T = 20     # first X timesteps to display
show_N = 8      # first Y neurons to display
max_neurons_plot = 4   # for line plots of u, show a few neurons

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

def compare_spikes(name, a, b):
    exact = np.array_equal(a, b)
    match_rate = (a == b).mean()
    print(f"{name}:")
    print(f"  exact match = {exact}")
    print(f"  match rate   = {match_rate * 100:.2f}%")

def ensure_2d(name, arr):
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")

def main():
    u_py = load_npy(PY_U)          # shape: (T, N)
    g_py = load_npy(PY_G)          # shape: (T, N)
    s_py = load_npy(PY_S)          # shape: (T, N)

    ensure_2d("u_py", u_py)
    ensure_2d("g_py", g_py)
    ensure_2d("s_py", s_py)

    T, N = u_py.shape

    # Loading C traces
    u_c = load_bin(C_U, np.float32, (T, N))
    g_c = load_bin(C_G, np.float32, (T, N))
    s_c = load_bin(C_S, np.uint8,   (T, N))

    # Loading GPU Traces
    u_gpu = load_bin(GPU_U, np.float32, (T, N))
    g_gpu = load_bin(GPU_G, np.float32, (T, N))
    s_gpu = load_bin(GPU_S, np.uint8,   (T, N))

    
    print("Shapes:")
    print("  Python:", u_py.shape)
    print("  C     :", u_c.shape)
    print("  GPU   :", u_gpu.shape)
    print()

    print("Comparing membrane potential u")
    du_py_c   = compare("Python vs C   (u)", u_py, u_c)
    du_py_gpu = compare("Python vs GPU (u)", u_py, u_gpu)
    du_c_gpu  = compare("C vs GPU       (u)", u_c, u_gpu)
    print()

    print("Comparing synaptic current g")
    dg_py_c   = compare("Python vs C   (g)", g_py, g_c)
    dg_py_gpu = compare("Python vs GPU (g)", g_py, g_gpu)
    dg_c_gpu  = compare("C vs GPU       (g)", g_c, g_gpu)
    print()

    print("Comparing spikes s")
    compare_spikes("Python vs C   (s)", s_py, s_c)
    compare_spikes("Python vs GPU (s)", s_py, s_gpu)
    compare_spikes("C vs GPU       (s)", s_c, s_gpu)
    print()

    # ----------------------------
    # Plot membrane potentials for a few neurons
    # ----------------------------
    t = np.arange(T)
    n_line = min(N, max_neurons_plot)

    fig, axes = plt.subplots(n_line, 1, figsize=(16, 3.2 * n_line), sharex=True)
    if n_line == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t[:show_T], u_py[:show_T, i], label="Python", linewidth=1.8)
        ax.plot(t[:show_T], u_c[:show_T, i], "--", label="C", linewidth=1.2)
        ax.plot(t[:show_T], u_gpu[:show_T, i], ":", label="GPU", linewidth=1.5)
        ax.set_ylabel(f"Neuron {i}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"Membrane potential comparison (first {show_T} timesteps)", y=1.01)
    fig.tight_layout()

    # ----------------------------
    # Absolute error plots for u
    # ----------------------------
    plt.figure(figsize=(16, 4))
    for i in range(min(N, 8)):
        plt.plot(t[:show_T], np.abs(du_py_c[:show_T, i]), label=f"n{i} Python-C")
    plt.xlabel("Time step")
    plt.ylabel("|u_py - u_c|")
    plt.title(f"Absolute error: Python vs C (first {show_T} timesteps)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()

    plt.figure(figsize=(16, 4))
    for i in range(min(N, 8)):
        plt.plot(t[:show_T], np.abs(du_py_gpu[:show_T, i]), label=f"n{i} Python-GPU")
    plt.xlabel("Time step")
    plt.ylabel("|u_py - u_gpu|")
    plt.title(f"Absolute error: Python vs GPU (first {show_T} timesteps)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()

    # ----------------------------
    # Spike raster comparison for first show_N neurons and first show_T timesteps
    # ----------------------------
    plt.figure(figsize=(16, 6))

    # Python
    py_times, py_neurons = np.where(s_py[:show_T, :show_N] > 0)
    plt.scatter(py_times, py_neurons, s=20, marker="|", label="Python spikes")

    # C
    c_times, c_neurons = np.where(s_c[:show_T, :show_N] > 0)
    plt.scatter(c_times, c_neurons, s=20, marker="x", label="C spikes")

    # GPU
    gpu_times, gpu_neurons = np.where(s_gpu[:show_T, :show_N] > 0)
    plt.scatter(gpu_times, gpu_neurons, s=20, marker=".", label="GPU spikes")

    plt.xlim(0, show_T)
    plt.ylim(-1, show_N)
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.title(f"Spike comparison: first {show_N} neurons, first {show_T} timesteps")
    plt.yticks(range(show_N))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()