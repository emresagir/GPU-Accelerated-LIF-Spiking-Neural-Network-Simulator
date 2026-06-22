import subprocess
import csv
import re
from pathlib import Path

# Sweep values
N_values = [256, 512, 1024, 2048, 4096, 8192, 16384]
T_values = [100, 500, 1000]

# Files
csv_filename = "benchmark_results_torch.csv"
reference_script = "snn_reference.py"
temp_script = "snn_reference_tmp.py"

# Parse output like:
# Completed 3000 steps, time = 0.1234s, ms/step = 0.0411
time_pattern = re.compile(r"time = ([0-9.]+)s, ms/step = ([0-9.]+)")

script_dir = Path(__file__).resolve().parent
ref_path = script_dir / reference_script
tmp_path = script_dir / temp_script
csv_path = script_dir / csv_filename

original_text = ref_path.read_text()

def make_patched_script(text: str, n: int, t: int) -> str:
    text = re.sub(
        r"^N\s*=\s*\d+\s*#\s*neurons\s*$",
        f"N = {n}           # neurons",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^T\s*=\s*\d+\s*#\s*timesteps\s*$",
        f"T = {t}           # timesteps",
        text,
        flags=re.MULTILINE,
    )
    return text

print("Starting Torch benchmark suite...")

try:
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # EXACT header format you want
        writer.writerow(["N", "T", "Time_ms"])

        for t in T_values:
            for n in N_values:

                print(f"Running N={n:5}, T={t:4}...", end=" ", flush=True)

                patched_text = make_patched_script(original_text, n, t)
                tmp_path.write_text(patched_text)

                run = subprocess.run(
                    ["python3", temp_script],
                    cwd=script_dir,
                    capture_output=True,
                    text=True,
                )

                if run.returncode != 0:
                    print("\n[ERROR] Run failed")
                    print(run.stderr)
                    continue

                match = time_pattern.search(run.stdout)
                if not match:
                    print("\n[ERROR] Could not parse runtime")
                    print(run.stdout)
                    continue

                time_s = float(match.group(1))
                time_ms = time_s * 1000.0   # ONLY ms

                # EXACT FORMAT: no formatting strings, no extra columns
                writer.writerow([n, t, f"{time_ms:.3f}"])

                print(f"Success ({time_ms:.3f} ms)")
finally:
    if tmp_path.exists():
        tmp_path.unlink()

print(f"\nBenchmarking complete! Results saved to {csv_filename}")