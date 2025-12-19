import subprocess
import os

# Generate ncu report for CW
subprocess.run(["python3", "src_py/sweep_cuda_only_ncu.py"], check=True)

# Generate ncu report for TW
subprocess.run(["python3", "src_py/sweep_tensor_only_ncu.py"], check=True)

# Generate ncu report for concurrent
subprocess.run(["python3", "src_py/sweep_concurrent_only_ncu.py"], check=True)

# Generate ncu utilization plots
subprocess.run(
    ["python3", "plot_concurrency_bars/plot_concurrency_bars.py"],
    check=True
)

# Generate runtime/speedup reports for serial vs concurrent (this already creates runtime/speedup plots)
subprocess.run(["python3", "src_py/sweep_serial_concurrent_no_ncu.py"], check=True)


