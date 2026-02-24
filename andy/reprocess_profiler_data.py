from profiler import Profiler
from pathlib import Path

# gpu_memory_total_gb so plots/reprocess use same y-axis when no live GPU (e.g. batch reprocess)
p = Profiler(
    output_directory="tests/runs",
    title="SLAM inference test",
    gpu_power_max_w=200,
    cpu_power_max_w=200,
    gpu_memory_total_gb=16.376,
)

# Reprocess a run: regenerate plot.png and metadata.json (with reference-adjusted stats if ref given)
p.reprocess_data("andy/runs/2026_02_23_1431_tartan_mono_offline_onnx_reproc")