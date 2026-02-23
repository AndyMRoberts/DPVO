#!/usr/bin/env python3
"""
Launcher for DPVO TartanAir evaluation (evaluate_tartan_andy.py).

Run from the project root. Creates a run directory andy/runs/YYYYMMDD_HHMM_<test_run_name>,
writes metadata with all parameters, then runs evaluate_tartan_andy.py with outputs directed there.

Optional profiling (GPU/power etc): use --power_log and the 'profiler' package
  (from profiler import Profiler). Run directory is created by the profiler.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

try:
    from profiler import Profiler
except ImportError:
    Profiler = None


def _format_with_stdev(value, stddev=None):
    """Format a numeric value with optional ± stddev on the same line."""
    if value is None:
        return "N/A"
    if stddev is not None and stddev is not False:
        try:
            s = float(stddev)
            if s == s:  # not nan
                return f"{value} ± {s}"
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_power_summary_txt(run_dir, meta):
    """
    Write power_summary.txt from profiler metadata.json content (meta dict).
    Uses ± stdev on the same line when _stddev is present.
    """
    txt_path = os.path.join(run_dir, "power_summary.txt")
    av = meta.get("averages") or {}
    ef = meta.get("energy_per_frame_j") or {}

    run_time_s = meta.get("run_time_s")
    num_frames = meta.get("num_frames")

    cpu_w = av.get("cpu_power_w")
    gpu_w = av.get("gpu_power_w")
    cpu_std = av.get("cpu_power_w_stddev")
    gpu_std = av.get("gpu_power_w_stddev")
    mean_power = (cpu_w or 0) + (gpu_w or 0)
    # Approximate stddev of sum (independent): sqrt(sigma_cpu^2 + sigma_gpu^2)
    mean_power_std = None
    if cpu_std is not None and gpu_std is not None:
        mean_power_std = (float(cpu_std) ** 2 + float(gpu_std) ** 2) ** 0.5
    elif cpu_std is not None:
        mean_power_std = float(cpu_std)
    elif gpu_std is not None:
        mean_power_std = float(gpu_std)

    total_energy_j = (run_time_s * mean_power) if (run_time_s and mean_power) else None
    total_energy_std = (run_time_s * mean_power_std) if (run_time_s and mean_power_std is not None) else None
    total_energy_kj = round(total_energy_j / 1000, 2) if total_energy_j is not None else None
    total_energy_kj_std = round(total_energy_std / 1000, 2) if total_energy_std else None

    gpu_mem_gb = av.get("gpu_memory_gb")
    gpu_mem_std = av.get("gpu_memory_gb_stddev")
    mean_gpu_memory_mib = round(gpu_mem_gb * 1024, 2) if gpu_mem_gb is not None else None
    mean_gpu_memory_mib_std = round(gpu_mem_std * 1024, 2) if gpu_mem_std is not None else None

    data_csv = os.path.join(run_dir, "data.csv")
    max_gpu_memory_gb = _max_gpu_memory_gb_from_csv(data_csv)
    max_gpu_memory_mib = round(max_gpu_memory_gb * 1024, 2) if max_gpu_memory_gb is not None else None

    ef_avg = ef.get("avg")
    ef_avg_std = ef.get("avg_stddev")
    ef_mj = round(ef_avg * 1000, 2) if ef_avg is not None else None
    ef_mj_std = round(ef_avg_std * 1000, 2) if ef_avg_std is not None else None

    lines = [
        "Power and timing summary (from profiler)",
        "=" * 50,
        "",
        f"run_duration_s: {_format_with_stdev(run_time_s)}",
        f"total_energy_J: {_format_with_stdev(total_energy_j, total_energy_std)}",
        f"total_energy_kJ: {_format_with_stdev(total_energy_kj, total_energy_kj_std)}",
        f"mean_power_W: {_format_with_stdev(round(mean_power, 2) if mean_power else None, round(mean_power_std, 2) if mean_power_std is not None else None)}",
        f"mean_gpu_memory_MiB: {_format_with_stdev(mean_gpu_memory_mib, mean_gpu_memory_mib_std)}",
        f"max_gpu_memory_MiB: {max_gpu_memory_mib if max_gpu_memory_mib is not None else 'N/A'}",
        f"total_frames: {num_frames if num_frames is not None else 'N/A'}",
        f"energy_per_frame_J: {_format_with_stdev(ef_avg, ef_avg_std)}",
        f"energy_per_frame_mJ: {_format_with_stdev(ef_mj, ef_mj_std)}",
    ]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return txt_path


def _max_gpu_memory_gb_from_csv(csv_path):
    """Read profiler data.csv and return max gpu_memory_gb (column index 5)."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            vals = []
            for row in reader:
                if len(row) > 5 and row[5].strip():
                    try:
                        vals.append(float(row[5]))
                    except ValueError:
                        pass
            return max(vals) if vals else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Launch DPVO TartanAir evaluation (evaluate_tartan_andy.py)"
    )
    parser.add_argument("--test_run_name", type=str, required=True,
                        help="Name for this test run (used in run directory)")
    parser.add_argument("--datapath", type=str, default=None,
                        help="Data root (overrides default TartanAir path)")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="Ground truth root (overrides default)")
    parser.add_argument("--weights", type=str, default="dpvo.pth")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--split", type=str, default="validation",
                        help="validation or test")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=18.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--show_img", action="store_true")
    parser.add_argument("--id", type=int, default=-1,
                        help="Single scene index (test split only); -1 = full evaluation")
    parser.add_argument("--opts", nargs="+", default=[],
                        help="Extra config options (key value pairs)")
    parser.add_argument("--power_log", action="store_true",
                        help="Profile CPU/GPU power and metrics during run. Requires 'profiler' package.")

    args = parser.parse_args()

    project_root = os.path.abspath(os.getcwd())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.test_run_name)
    runs_base = os.path.join(project_root, "andy", "runs")
    os.makedirs(runs_base, exist_ok=True)

    profiler_instance = None
    if args.power_log:
        if Profiler is None:
            print("Error: --power_log requires the 'profiler' package. Install it or add it to PYTHONPATH.", file=sys.stderr)
            sys.exit(1)
        Profiler.verify_setup() # ensures all data can be accessed before running
        profiler_instance = Profiler(runs_base, frequency_hz=2.0, title=safe_name,  cpu_power_max_w=150.0, gpu_power_max_w=200.0)
        ref_dir = os.path.join(runs_base, "reference")
        use_reference = os.path.isdir(ref_dir)
        run_dir = profiler_instance.start(use_reference=use_reference)
        print(f"Profiler started. Run directory: {run_dir}")
    else:
        run_dirname = f"{timestamp}_{safe_name}"
        run_dir = os.path.join(runs_base, run_dirname)
        os.makedirs(run_dir, exist_ok=True)

    params = {
        "test_run_name": args.test_run_name,
        "run_dir": run_dir,
        "datapath": args.datapath,
        "gt_path": args.gt_path,
        "weights": args.weights,
        "config": args.config,
        "split": args.split,
        "trials": args.trials,
        "backend_thresh": args.backend_thresh,
        "plot": args.plot,
        "save_trajectory": args.save_trajectory,
        "viz": args.viz,
        "show_img": args.show_img,
        "scene_id": args.id,
        "opts": args.opts,
        "timestamp": timestamp,
    }

    metadata_txt_path = os.path.join(run_dir, "metadata.txt")
    with open(metadata_txt_path, "w") as f:
        f.write("DPVO TartanAir evaluation run metadata\n")
        f.write("=" * 60 + "\n\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

    eval_script = os.path.join(project_root, "evaluate_tartan_andy.py")
    cmd = [
        sys.executable, eval_script,
        "--run_dir", run_dir,
        "--weights", args.weights,
        "--config", args.config,
        "--split", args.split,
        "--trials", str(args.trials),
        "--backend_thresh", str(args.backend_thresh),
        "--id", str(args.id),
    ]
    if args.datapath:
        cmd.extend(["--datapath", args.datapath])
    if args.gt_path:
        cmd.extend(["--gt_path", args.gt_path])
    if args.plot:
        cmd.append("--plot")
    if args.save_trajectory:
        cmd.append("--save_trajectory")
    if args.viz:
        cmd.append("--viz")
    if args.show_img:
        cmd.append("--show_img")
    if args.opts:
        cmd.extend(["--opts"] + args.opts)

    print(f"Run directory: {run_dir}")
    print(f"Metadata written to {metadata_txt_path}")
    print("Launching evaluate_tartan_andy.py...")
    print(" ".join(cmd))

    result = subprocess.run(cmd, cwd=project_root)

    total_frames = None
    ate_path = os.path.join(run_dir, "ate_results.json")
    if os.path.isfile(ate_path):
        with open(ate_path) as f:
            ate_data = json.load(f)
        total_frames = ate_data.get("total_frames")
    
    

    if profiler_instance is not None:
        profiler_instance.stop(num_frames=total_frames)
        print("Profiler stopped. Power log and plot written by profiler (data.csv, plot.png).")

        meta_path = os.path.join(run_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["launch_params"] = params
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            with open(metadata_txt_path, "a") as f:
                f.write("(Profiler data: metadata.json; human-readable power summary: power_summary.txt)\n")
            power_summary_txt = _write_power_summary_txt(run_dir, meta)
            print(f"Power summary (human-readable) saved to {power_summary_txt}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
