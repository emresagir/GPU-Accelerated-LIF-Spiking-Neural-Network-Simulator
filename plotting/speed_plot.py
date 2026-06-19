import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the benchmark data from the CSV files
df_basic = pd.read_csv("../gpu_version/benchmark_results_basic.csv")
df_optimized = pd.read_csv("../optimized_gpu_version/benchmark_results_optimized.csv")

# ==========================================
# CHANGE THIS VALUE TO CHOOSE YOUR T STEPS
# Options from your data: 100, 500, or 1000
# ==========================================
CHOSEN_T = 1000

# 2. Filter datasets to isolate only the chosen T value
df_basic_filtered = df_basic[df_basic['T'] == CHOSEN_T]
df_optimized_filtered = df_optimized[df_optimized['T'] == CHOSEN_T]

# 3. Merge the datasets on N to calculate the speedup factor
df_merged = pd.merge(df_basic_filtered, df_optimized_filtered, on='N', suffixes=('_basic', '_optimized'))
df_merged['Speedup'] = df_merged['Time_ms_basic'] / df_merged['Time_ms_optimized']


# =========================================================================
# FILE 1: EXECUTION TIMES GRAPH (SPEED VALUES)
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(8, 5))

# Plot Basic implementation (Solid blue line with circular markers)
ax1.plot(df_basic_filtered['N'], df_basic_filtered['Time_ms'], marker='o', linestyle='-', 
        color='tab:blue', linewidth=2, label=f'Basic (T={CHOSEN_T})')

# Plot Optimized implementation (Dashed orange line with square markers)
ax1.plot(df_optimized_filtered['N'], df_optimized_filtered['Time_ms'], marker='s', linestyle='--', 
        color='tab:orange', linewidth=2, label=f'Optimized (T={CHOSEN_T})')

# Apply scales: X is log(2) because N doubles; Y is normal (linear) 
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')

# Explicitly mark your test boundaries along the X axis
ax1.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384])
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Add structural labels and title
ax1.set_xlabel('Number of Neurons (N)')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title(f'SNN Simulation Performance Scaling for T = {CHOSEN_T} (Log Y-Axis)')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()
fig1.tight_layout()

# Save the first plot
fig1.savefig('../report_ieee_v/images/snn_execution_times.png', dpi=300)


# =========================================================================
# FILE 2: SPEEDUP FACTOR GRAPH
# =========================================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))

# Plot the Speedup factor curve (Solid green line with triangular markers)
ax2.plot(df_merged['N'], df_merged['Speedup'], marker='^', linestyle='-', 
        color='tab:green', linewidth=2, label=f'Speedup Factor (T={CHOSEN_T})')

# Apply scales: X is log(2) because N doubles; Y is normal (linear)
ax2.set_xscale('log', base=2)
ax2.set_yscale('linear')

# Explicitly mark your test boundaries along the X axis
ax2.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384])
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Add a baseline threshold line at y=1.0 (indicating equal performance)
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even (Speedup = 1x)')

# Add structural labels and title
ax2.set_xlabel('Number of Neurons (N)')
ax2.set_ylabel('Speedup Factor (x times faster)')
ax2.set_title(f'Achieved Speedup Factor for T = {CHOSEN_T}')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()
fig2.tight_layout()

# Save the second plot
fig2.savefig('../report_ieee_v/images/snn_speedup_factor.png', dpi=300)

print("Both plots generated and saved successfully!")