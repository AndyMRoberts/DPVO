#!/usr/bin/env python3
"""
Run DPVO TartanAir evaluation under cProfile and generate a pie chart
showing which DPVO modules consume the most cumulative time.

Usage example (from project root):

  python andy/profile_dpvo_cprofile.py \
      --test_run_name tartan_mono_offline_profile \
      --datapath datasets/TartanAir \
      --gt_path datasets/TartanAir \
      --weights dpvo.pth \
      --config config/default.yaml \
      --backend pytorch

This script mirrors the core arguments of evaluate_tartan_andy.py, runs
its full evaluation path (id < 0) under cProfile, and then:
  - writes a .prof stats file
  - aggregates cumulative time by DPVO module
  - saves a pie chart PNG of per-module time share
"""

import argparse
import cProfile
import datetime
import os
import os.path as osp
import pstats
from typing import Dict

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _make_run_dir(project_root: str, test_run_name: str or None) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    safe_name = (
        "".join(c if c.isalnum() or c in "-_" else "_" for c in test_run_name)
        if test_run_name
        else "dpvo_profile"
    )
    runs_base = osp.join(project_root, "andy", "runs")
    os.makedirs(runs_base, exist_ok=True)
    run_dirname = f"{timestamp}_{safe_name}"
    run_dir = osp.join(runs_base, run_dirname)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _aggregate_dpvo_by_module(stats: pstats.Stats, project_root: str) -> Dict[str, float]:
    """
    Aggregate cumulative time (ct) per DPVO module based on cProfile stats.
    """
    dpvo_root = osp.join(project_root, "dpvo")
    dpvo_root_norm = osp.normpath(dpvo_root)

    module_totals: Dict[str, float] = {}

    for func, stat in stats.stats.items():
        filename, _lineno, _funcname = func
        cc, nc, tt, ct, _callers = stat  # tt: total time, ct: cumulative time

        if not filename:
            continue

        filename_abs = osp.abspath(filename)
        if dpvo_root_norm not in filename_abs:
            continue

        # Use path relative to project root for labeling
        rel = osp.relpath(filename_abs, project_root)
        module_totals[rel] = module_totals.get(rel, 0.0) + ct

    return module_totals


def _make_pie_chart(module_totals: Dict[str, float], out_path: str, title: str) -> None:
    if not module_totals:
        print("No DPVO entries found in profile; skipping pie chart.")
        return

    if plt is None:
        print("matplotlib is not available; skipping pie chart.")
        return

    # Sort modules by cumulative time (descending)
    items = sorted(module_totals.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    sizes = [v for _, v in items]

    total = sum(sizes)
    if total <= 0:
        print("Total DPVO time is zero; skipping pie chart.")
        return

    # Limit to top N and group rest as "other" to keep chart readable
    top_n = 8
    if len(sizes) > top_n:
        top_labels = labels[:top_n]
        top_sizes = sizes[:top_n]
        other_size = sum(sizes[top_n:])
        top_labels.append("other")
        top_sizes.append(other_size)
        labels, sizes = top_labels, top_sizes

    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%"
        if pct >= 1
        else "",  # hide tiny slices' labels
        startangle=140,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved DPVO time breakdown pie chart to {out_path}")


def main() -> None:
    project_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(
        description="Profile DPVO TartanAir evaluation with cProfile and produce a pie chart."
    )
    parser.add_argument(
        "--test_run_name",
        type=str,
        default=None,
        help="Name for this profiling run (used in andy/runs directory).",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help="Data root (overrides default TartanAir path).",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Ground truth root (overrides default).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="dpvo.pth",
        help="Path to DPVO weights file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to DPVO config YAML.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split: validation or test.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of evaluation trials per scene.",
    )
    parser.add_argument(
        "--backend_thresh",
        type=float,
        default=18.0,
        help="Backend threshold value (forwarded to cfg.BACKEND_THRESH).",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx"],
        default="pytorch",
        help="Run with pure PyTorch or PyTorch+ONNX (encoders via ONNX).",
    )
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default="andy/onnx",
        help="Directory containing fnet.onnx and inet.onnx (used when --backend onnx).",
    )
    parser.add_argument(
        "--opts",
        nargs="+",
        default=[],
        help="Extra config options (key value pairs) to merge into cfg.",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=-1,
        help="Single scene index (test split only); -1 = full evaluation. "
        "Profiling is intended for the full evaluation path (id < 0).",
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default=None,
        help="Path for cProfile stats (.prof). Defaults to <run_dir>/dpvo_profile.prof.",
    )
    parser.add_argument(
        "--pie_output",
        type=str,
        default=None,
        help="Path for pie chart PNG. Defaults to <run_dir>/dpvo_profile_pie.png.",
    )

    args = parser.parse_args()

    # Lazily import evaluate_tartan_andy only after paths are set
    import sys

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from evaluate_tartan_andy import cfg, evaluate, run  # type: ignore

    if args.id >= 0:
        raise SystemExit(
            "This profiler script is designed for the full evaluation path (id < 0). "
            "Run with --id -1 (default) for an aggregate evaluation."
        )

    run_dir = _make_run_dir(project_root, args.test_run_name)

    prof_path = args.profile_output or osp.join(run_dir, "dpvo_profile.prof")
    pie_path = args.pie_output or osp.join(run_dir, "dpvo_profile_pie.png")

    # Configure cfg as in evaluate_tartan_andy.__main__
    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    onnx_dir = os.path.abspath(args.onnx_dir) if args.backend == "onnx" else None
    if onnx_dir and not os.path.isdir(onnx_dir):
        raise FileNotFoundError(
            f"ONNX dir not found: {onnx_dir}. Run andy/onnx_conversion.ipynb and set --onnx_dir."
        )

    # Match evaluate_tartan_andy.py expectations: it always writes a small
    # text file into <run_dir>/trajectory_data, even when plot/save are False.
    traj_dir = osp.join(run_dir, "trajectory_data")
    os.makedirs(traj_dir, exist_ok=True)

    print("Running DPVO evaluation under cProfile with config:")
    print(cfg)
    print("Backend:", args.backend, f"(onnx_dir={onnx_dir})" if onnx_dir else "")
    print(f"Run directory: {run_dir}")
    print(f"cProfile stats will be saved to: {prof_path}")
    print(f"Pie chart (per-module DPVO time) will be saved to: {pie_path}")

    prof = cProfile.Profile()

    def _evaluate():
        # For full evaluation (id < 0) evaluate_tartan_andy.py calls evaluate()
        return evaluate(
            cfg,
            args.weights,
            split=args.split,
            trials=args.trials,
            plot=False,
            save=False,
            run_dir=run_dir,
            datapath=args.datapath,
            gt_path=args.gt_path,
            onnx_dir=onnx_dir,
        )

    # Run evaluation under profiler
    results = prof.runcall(_evaluate)

    # Save raw stats
    prof.dump_stats(prof_path)
    print("cProfile run complete.")

    # Build Stats object for aggregation
    stats = pstats.Stats(prof)
    stats.sort_stats("cumtime")

    module_totals = _aggregate_dpvo_by_module(stats, project_root)
    if module_totals:
        print("Top DPVO modules by cumulative time:")
        for module, t in sorted(
            module_totals.items(), key=lambda kv: kv[1], reverse=True
        )[:15]:
            print(f"  {module}: {t:.3f} s")
    else:
        print("No DPVO entries found in cProfile stats.")

    # Generate pie chart
    _make_pie_chart(
        module_totals,
        pie_path,
        title="DPVO cumulative time by module (cProfile cumtime)",
    )

    # Echo evaluation results at the end for convenience
    print("Evaluation results (ATE metrics):")
    print(results)


if __name__ == "__main__":
    main()

