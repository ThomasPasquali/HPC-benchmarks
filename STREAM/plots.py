#!/usr/bin/env python3
"""
plot_stream.py — Visualise STREAM benchmark results from **files** *or* **sbm jobs**.

* Two sub-commands:
  * **files** - ORIGINAL workflow (one or more plain-text STREAM outputs).
  * **sbm**   - Pulls completed jobs via ``sbm.jobs_list`` and parses their
    captured *stdout* on the fly (see example below).
* A shared plotting backend, so both paths end up on the same figure.

Example - classic file mode
───────────────────────────
```bash
python plot_stream.py files A100_1c.txt A100_32c.txt \
                         -H A100 A100 -o stream.png
```

Example - sbm mode (no filenames needed!)
─────────────────────────────────────────
```bash
python plot_stream.py sbm -s COMPLETE -o stream.png
```
This walks over
```python
jobs = sbm.jobs_list(from_active=True, from_archived=True,
                     status=["COMPLETE"])
```
then, for every job whose ``config_name`` looks like
``HWName_16cpus`` (regex ``(\w+)_(\d+)cpus``), it extracts the STREAM metrics
straight from ``job.get_stdout()``.

Dependencies
────────────
Python ≥3.9, pandas, matplotlib, and of course your in-house ``sbm`` package.
Install the PyPI bits with: ``pip install pandas matplotlib``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns and constants
# ──────────────────────────────────────────────────────────────────────────────
FUNCTIONS = ["Copy", "Scale", "Add", "Triad"]
_RATE_RE = re.compile(rf"^({'|'.join(FUNCTIONS)}):\s+([0-9]+(?:\.[0-9]+)?)")
_THREADS_RE = re.compile(r"Number of Threads counted\s*=\s*(\d+)")
_JOB_RE = re.compile(r"(\w+)_(\d+)cpus")  # captures hw and core count

# ──────────────────────────────────────────────────────────────────────────────
# Low-level parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_metrics(lines: List[str]) -> dict[str, float]:
  """Return a dict with keys 'cores', 'Copy', 'Scale', 'Add', 'Triad'."""
  metrics: dict[str, float] = {}
  for line in lines:
    if "cores" not in metrics:
      mt = _THREADS_RE.search(line)
      if mt:
        metrics["cores"] = int(mt.group(1))

    mk = _RATE_RE.match(line.strip())
    if mk:
      func, rate = mk.groups()
      metrics[func] = float(rate)

    if len(metrics) == 1 + len(FUNCTIONS):
      break  # early exit once all fields are present

  missing = [k for k in (FUNCTIONS + ["cores"]) if k not in metrics]
  if missing:
    raise ValueError(f"missing fields: {', '.join(missing)}")
  return metrics


def _parse_stream_text(text: str) -> dict[str, float]:
  return _parse_metrics(text.splitlines())


def _parse_single_stream_file(path: str) -> dict[str, float]:
  with open(path, "r", encoding="utf-8", errors="ignore") as fh:
    return _parse_metrics(fh)

# ──────────────────────────────────────────────────────────────────────────────
# DataFrame builders
# ──────────────────────────────────────────────────────────────────────────────

def _infer_hardware_labels(paths: List[str], manual: List[str] | None) -> List[str]:
  if manual is None:
    return [os.path.basename(p).split("_")[0] for p in paths]
  if len(manual) != len(paths):
    raise ValueError("--hardware/-H labels must match number of input files")
  return manual


def _build_dataframe_from_files(paths: List[str], hw_labels: List[str]) -> pd.DataFrame:
  rows = []
  for path, hw in zip(paths, hw_labels):
    parsed = _parse_single_stream_file(path)
    for func in FUNCTIONS:
      rows.append({
        "hardware": hw,
        "cores": parsed["cores"],
        "function": func,
        "bandwidth_MBps": parsed[func],
      })
  return pd.DataFrame(rows).sort_values(["function", "hardware", "cores"])


def _build_dataframe_from_jobs(status: List[str]) -> pd.DataFrame:
  import sbatchman as sbm

  jobs = sbm.jobs_list(from_active=True, from_archived=True, status=[sbm.Status[s] for s in status])
  rows: list[dict] = []
  for job in jobs:
    m = _JOB_RE.match(job.config_name)
    if not m:
      continue  # skip unrelated jobs
    hw, cores_str = m.groups()
    cores = int(cores_str)
    stdout = job.get_stdout()
    try:
      parsed = _parse_stream_text(stdout)
    except ValueError as exc:
      print(f"[WARN] job {job.id}: {exc}", file=sys.stderr)
      continue
    for func in FUNCTIONS:
      rows.append({
        "hardware": hw,
        "cores": cores,
        "function": func,
        "bandwidth_MBps": parsed[func],
      })

  if not rows:
    raise RuntimeError("No valid STREAM outputs found via sbm.")
  return pd.DataFrame(rows).sort_values(["function", "hardware", "cores"])

# ──────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ──────────────────────────────────────────────────────────────────────────────

def _plot(df: pd.DataFrame, output: str | None) -> None:
  marker_cycle = ["o", "s", "^", "d", "x", "P", "*", "v", ">"]

  fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
  axes = axes.flatten()

  for idx, func in enumerate(FUNCTIONS):
    ax = axes[idx]
    func_df = df[df["function"] == func]
    for j, (hw, group) in enumerate(func_df.groupby("hardware", sort=False)):
      group_sorted = group.sort_values("cores")
      ax.plot(
        group_sorted["cores"],
        group_sorted["bandwidth_MBps"],
        label=hw,
        marker=marker_cycle[j % len(marker_cycle)],
        linewidth=1.8,
      )
    ax.set_title(func)
    ax.set_xlabel("CPU cores")
    ax.set_ylabel("Bandwidth [MB/s]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize="small")

  fig.suptitle("STREAM - Memory-Bandwidth Scaling", fontsize=17, y=0.97)
  fig.tight_layout(rect=[0, 0, 1, 0.95])

  if output:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    print(f"[ OK ] figure saved to {output}")
  else:
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Unified CLI (sub-commands: files / sbm)
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot STREAM results from plain files **or** directly from SbatchMan jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── files sub-command ────────────────────────────────────────────────
    files_p = subparsers.add_parser("files", help="Parse one or more STREAM output files")
    files_p.add_argument("inputs", nargs="+", help="STREAM output text files")
    files_p.add_argument("-H", "--hardware", nargs="+", help="Hardware label per input file")
    files_p.add_argument("-o", "--output", help="Save figure instead of displaying it")

    # ── sbm sub-command ────────────────────────────────────────────────
    sbm_p = subparsers.add_parser("sbm", help="Pull STREAM outputs from sbm jobs")
    sbm_p.add_argument("-s", "--status", nargs="+", default=["COMPLETE"], help="Job status filter")
    sbm_p.add_argument("-o", "--output", help="Save figure instead of displaying it")

    args = parser.parse_args(argv)

    if args.mode == "files":
        labels = _infer_hardware_labels(args.inputs, args.hardware)
        df = _build_dataframe_from_files(args.inputs, labels)
        _plot(df, args.output)

    elif args.mode == "sbm":
        df = _build_dataframe_from_jobs(args.status)
        _plot(df, args.output)


if __name__ == "__main__":
    main()

# Run example: python3 plots.py sbm -s COMPLETED -o results/stream.png