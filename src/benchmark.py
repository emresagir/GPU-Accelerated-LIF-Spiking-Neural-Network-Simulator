import subprocess
import csv
import re
import os

# Parameter sweep
N_values = [256, 512, 1024, 2048, 4096, 8192, 16384]
T_values = [100, 500, 1000]

# Paths
c_source = "cpu_model.c"
executable = "./sim_exec"
csv_filename = "benchmark_results_cpu.csv"

# Match:
# "The simulation took, <seconds>.<nanoseconds>"
time_pattern = re.compile(r"The simulation took, (\d+)\.(\d+)")

# Ensure data directory exists
os.makedirs("../torch", exist_ok=True)

print("Starting benchmark suite...")

with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["N", "T", "Time_ms"])

    for t in T_values:
        for n in N_values:

            print(f"Generating data, compiling and running N={n}, T={t}...", end=" ", flush=True)

            #
            # 1. Generate dummy input files in ../torch
            #
            gen_process = subprocess.run(
                [
                    "python3",
                    "-c",
                    f"""
import numpy as np

np.random.seed(1234)

W = (np.random.rand({n}, {n}).astype(np.float32) * 0.05)
W.tofile('../torch/W_post_pre.f32')

ext = (np.random.rand({t}, {n}) < 0.02).astype(np.uint8)
ext.tofile('../torch/ext_spikes.u8')
"""
                ],
                capture_output=True,
                text=True,
            )

            if gen_process.returncode != 0:
                print("\n[ERROR] Failed to generate input files:")
                print(gen_process.stderr)
                continue

            #
            # 2. Compile with N and T macros
            #
            compile_cmd = [
                "gcc",
                "-Wall",
                "-Wextra",
                "-O3",
                f"-DN={n}",
                f"-DT={t}",
                "cpu_model.c",
                "utilities.c",
                "-o",
                executable,
                "-lm",
            ]

            compile_process = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True
            )

            if compile_process.returncode != 0:
                print(f"\n[ERROR] Compilation failed for N={n}, T={t}:")
                print(compile_process.stderr)
                continue

            #
            # 3. Run executable
            #
            run_process = subprocess.run(
                [executable],
                capture_output=True,
                text=True
            )

            if run_process.returncode != 0:
                print(f"\n[ERROR] Execution failed for N={n}, T={t}:")
                print(run_process.stderr)
                continue

            #
            # 4. Parse execution time
            #
            output = run_process.stdout
            match = time_pattern.search(output)

            if match:
                sec = int(match.group(1))
                nsec = int(match.group(2))

                time_ms = (sec * 1000.0) + (nsec / 1_000_000.0)

                writer.writerow([n, t, f"{time_ms:.3f}"])

                print(f"Success ({time_ms:.3f} ms)")
            else:
                print("\n[ERROR] Failed to parse execution time.")
                print("Program output:")
                print(output)

# Cleanup
if os.path.exists(executable):
    os.remove(executable)

print(f"\nBenchmarking complete! Results saved to {csv_filename}")