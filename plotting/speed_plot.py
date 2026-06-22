import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# =============================================================================
# LOAD ALL BENCHMARK DATA
# =============================================================================
df_basic     = pd.read_csv("../gpu_version/benchmark_results_basic.csv")
df_optimized = pd.read_csv("../optimized_gpu_version/benchmark_results_optimized.csv")
df_cpu       = pd.read_csv("../src/benchmark_results_cpu.csv")
df_torch     = pd.read_csv("../torch/benchmark_results_torch.csv")

# Assign readable labels and colors used consistently across every figure.
# Changing here propagates everywhere.
IMPL = {
    "CPU":           {"df": df_cpu,       "color": "tab:gray",   "marker": "D", "ls": "-"},
    "Torch":         {"df": df_torch,     "color": "tab:purple", "marker": "v", "ls": "-."},
    "Basic GPU":     {"df": df_basic,     "color": "tab:blue",   "marker": "o", "ls": "-"},
    "Optimized GPU": {"df": df_optimized, "color": "tab:orange", "marker": "s", "ls": "--"},
}

# ==========================================
# CHANGE THIS VALUE TO CHOOSE YOUR T STEPS
# Options from your data: 100, 500, or 1000
# ==========================================
CHOSEN_T = 1000

ALL_T = [100, 500, 1000]   # used in multi-T figures below
X_TICKS = [256, 512, 1024, 2048, 4096, 8192, 16384]

OUTPUT_DIR = "../report_ieee_v/images/"


# =============================================================================
# HELPER: filter a dataframe to a single T value
# =============================================================================
def filter_t(df, t):
    return df[df["T"] == t].copy()


# =============================================================================
# ORIGINAL FIGURE 1  ── Execution times, Basic GPU vs Optimized GPU
# =============================================================================

df_basic_f     = filter_t(df_basic,     CHOSEN_T)
df_optimized_f = filter_t(df_optimized, CHOSEN_T)

fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(df_basic_f["N"],     df_basic_f["Time_ms"],     marker="o", linestyle="-",
         color="tab:blue",   linewidth=2, label=f"Basic GPU (T={CHOSEN_T})")
ax1.plot(df_optimized_f["N"], df_optimized_f["Time_ms"], marker="s", linestyle="--",
         color="tab:orange", linewidth=2, label=f"Optimized GPU (T={CHOSEN_T})")

ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xticks(X_TICKS)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel("Number of Neurons (N)")
ax1.set_ylabel("Execution Time (ms)")
ax1.set_title(f"SNN Simulation Performance Scaling for T = {CHOSEN_T} (Log Y-Axis)")
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR + "snn_execution_times.png", dpi=300)


# =============================================================================
# ORIGINAL FIGURE 2  ── Speedup: Optimized GPU vs Basic GPU 
# =============================================================================
df_merged_orig = pd.merge(df_basic_f, df_optimized_f, on="N",
                          suffixes=("_basic", "_optimized"))
df_merged_orig["Speedup"] = (df_merged_orig["Time_ms_basic"] /
                              df_merged_orig["Time_ms_optimized"])

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(df_merged_orig["N"], df_merged_orig["Speedup"], marker="^", linestyle="-",
         color="tab:green", linewidth=2, label=f"Speedup Factor (T={CHOSEN_T})")

ax2.set_xscale("log", base=2)
ax2.set_xticks(X_TICKS)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Break-even (Speedup = 1x)")
ax2.set_xlabel("Number of Neurons (N)")
ax2.set_ylabel("Speedup Factor (x times faster)")
ax2.set_title(f"Achieved Speedup Factor for T = {CHOSEN_T}")
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR + "snn_speedup_factor.png", dpi=300)


# =============================================================================
# NEW FIGURE 3  ── All-4 execution times on one plot
# =============================================================================

fig3, ax3 = plt.subplots(figsize=(9, 5))

for label, meta in IMPL.items():
    df_f = filter_t(meta["df"], CHOSEN_T)
    ax3.plot(df_f["N"], df_f["Time_ms"],
             marker=meta["marker"], linestyle=meta["ls"],
             color=meta["color"], linewidth=2, label=label)

ax3.set_xscale("log", base=2)
ax3.set_yscale("log")
ax3.set_xticks(X_TICKS)
ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax3.set_xlabel("Number of Neurons (N)")
ax3.set_ylabel("Execution Time (ms)")
ax3.set_title(f"All Implementations — Execution Time Scaling  (T={CHOSEN_T}, Log-Log)")
ax3.grid(True, which="both", ls="--", alpha=0.5)
ax3.legend()
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR + "snn_all_exec_times.png", dpi=300)


# =============================================================================
# NEW FIGURE 4  ── Speedup of all implementations normalised to CPU baseline
# =============================================================================


df_cpu_f = filter_t(df_cpu, CHOSEN_T).rename(columns={"Time_ms": "Time_ms_cpu"})

fig4, ax4 = plt.subplots(figsize=(9, 5))

for label, meta in IMPL.items():
    if label == "CPU":
        continue   # skip the baseline itself
    df_f = filter_t(meta["df"], CHOSEN_T)
    merged = pd.merge(df_cpu_f[["N", "Time_ms_cpu"]], df_f, on="N")
    merged["Speedup_vs_CPU"] = merged["Time_ms_cpu"] / merged["Time_ms"]
    ax4.plot(merged["N"], merged["Speedup_vs_CPU"],
             marker=meta["marker"], linestyle=meta["ls"],
             color=meta["color"], linewidth=2, label=label)

ax4.set_xscale("log", base=2)
ax4.set_xticks(X_TICKS)
ax4.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax4.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="CPU Baseline")
ax4.set_xlabel("Number of Neurons (N)")
ax4.set_ylabel("Speedup vs CPU (×)")
ax4.set_title(f"Speedup Relative to CPU Baseline  (T={CHOSEN_T})")
ax4.grid(True, which="both", ls="--", alpha=0.5)
ax4.legend()
fig4.tight_layout()
fig4.savefig(OUTPUT_DIR + "snn_speedup_vs_cpu.png", dpi=300)


# =============================================================================
# NEW FIGURE 5  ── Bar chart at fixed N — all implementations × all T values
# =============================================================================
FIXED_N = 16384   # change to your largest tested N

fig5, ax5 = plt.subplots(figsize=(10, 5))

n_impls = len(IMPL)
n_t     = len(ALL_T)
bar_w   = 0.18
x       = np.arange(n_t)   # one group per T value

for i, (label, meta) in enumerate(IMPL.items()):
    times = []
    for t in ALL_T:
        row = meta["df"][(meta["df"]["T"] == t) & (meta["df"]["N"] == FIXED_N)]
        times.append(row["Time_ms"].values[0] if len(row) else float("nan"))
    offset = (i - n_impls / 2 + 0.5) * bar_w
    ax5.bar(x + offset, times, width=bar_w,
            color=meta["color"], label=label, alpha=0.85, edgecolor="black", linewidth=0.5)

ax5.set_yscale("log")
ax5.set_xticks(x)
ax5.set_xticklabels([f"T={t}" for t in ALL_T])
ax5.set_xlabel("Timesteps (T)")
ax5.set_ylabel("Execution Time (ms, log scale)")
ax5.set_title(f"Execution Time at N={FIXED_N} — All Implementations & Timesteps")
ax5.legend()
ax5.grid(True, axis="y", ls="--", alpha=0.5)
fig5.tight_layout()
fig5.savefig(OUTPUT_DIR + "snn_bar_fixed_n.png", dpi=300)


# =============================================================================
# NEW FIGURE 6  ── Speedup heatmap across all T values (normalised to CPU)
# =============================================================================

non_cpu_labels = [k for k in IMPL if k != "CPU"]
fig6, axes6 = plt.subplots(1, len(non_cpu_labels),
                            figsize=(5 * len(non_cpu_labels), 5),
                            sharey=True)

N_vals = sorted(df_cpu["N"].unique())

for ax, label in zip(axes6, non_cpu_labels):
    meta  = IMPL[label]
    matrix = np.full((len(N_vals), len(ALL_T)), np.nan)

    for j, t in enumerate(ALL_T):
        cpu_f  = filter_t(df_cpu,      t).set_index("N")["Time_ms"]
        impl_f = filter_t(meta["df"],  t).set_index("N")["Time_ms"]
        for i, n in enumerate(N_vals):
            if n in cpu_f.index and n in impl_f.index:
                matrix[i, j] = cpu_f[n] / impl_f[n]

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0.5, vmax=matrix[~np.isnan(matrix)].max())

    # Annotate each cell with the numeric speedup
    for i in range(len(N_vals)):
        for j in range(len(ALL_T)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}×",
                        ha="center", va="center", fontsize=8,
                        color="black" if 0.4 < matrix[i, j] / matrix[~np.isnan(matrix)].max() < 0.85
                        else "white")

    ax.set_xticks(range(len(ALL_T)))
    ax.set_xticklabels([f"T={t}" for t in ALL_T])
    ax.set_yticks(range(len(N_vals)))
    ax.set_yticklabels(N_vals)
    ax.set_xlabel("Timesteps (T)")
    ax.set_title(f"{label} vs CPU")
    plt.colorbar(im, ax=ax, label="Speedup (×)")

if len(non_cpu_labels) > 0:
    axes6[0].set_ylabel("Number of Neurons (N)")

fig6.suptitle("Speedup Heatmap vs CPU — All Implementations", fontsize=13, y=1.02)
fig6.tight_layout()
fig6.savefig(OUTPUT_DIR + "snn_speedup_heatmap.png", dpi=300, bbox_inches="tight")


print("All 6 plots generated and saved successfully!")
print(f"  Fig 1 — Original: Basic vs Optimized GPU execution times  (T={CHOSEN_T})")
print(f"  Fig 2 — Original: Basic vs Optimized GPU speedup          (T={CHOSEN_T})")
print(f"  Fig 3 — NEW ★★★★★: All 4 implementations, execution time  (T={CHOSEN_T})")
print(f"  Fig 4 — NEW ★★★★☆: Speedup vs CPU baseline               (T={CHOSEN_T})")
print(f"  Fig 5 — NEW ★★★★☆: Bar chart at N={FIXED_N}, all T values")
print(f"  Fig 6 — NEW ★★★☆☆: Speedup heatmap (appendix-level detail)")