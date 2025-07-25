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
python plot_stream.py files A100_1c.txt A100_32c.txt -H A100 A100
```

Example - sbm mode (no filenames needed!)
─────────────────────────────────────────
```bash
python plot_stream.py sbm -s COMPLETE
```
This walks over
```python
jobs = sbm.jobs_list(from_active=True, from_archived=True,
                     status=["COMPLETE"])
```
then, for every job whose ``config_name`` looks like
``HWName_16cpus`` (regex ``(\\w+)_(\\d+)cpus``), it extracts the STREAM metrics
straight from ``job.get_stdout()``.

Dependencies
────────────
Python ≥3.9, pandas, matplotlib, and of course your in-house ``sbm`` package.
Install the PyPI bits with: ``pip install pandas matplotlib``.
"""

import argparse
import itertools
import os
from pathlib import Path
import re
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns and constants
# ──────────────────────────────────────────────────────────────────────────────

FUNCTIONS = ["Copy", "Scale", "Add", "Triad"]
_RATE_RE = re.compile(rf"^({'|'.join(FUNCTIONS)}):\s+([0-9]+(?:\.[0-9]+)?)")
_THREADS_RE = re.compile(r"Number of Threads counted\s*=\s*(\d+)")
_JOB_RE = re.compile(r"(\w+)_(\d+)cpus")  # captures hw and core count


OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


FONT_TITLE = 38
FONT_AXES = 28
FONT_TICKS = 20
FONT_LEGEND = 16

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

BOARD_NAMES_MAP = {
  'brah': 'AMD EPYC 7742',
  'baldo': 'AMD EPYC 7742',
  'pioneer': 'Milk-V Pioneer',
  'bananaf3': 'Banana Pi F3',
  'arriesgado': 'HiFive Unmatched',
}

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

  jobs = sbm.jobs_list(from_active=True, from_archived=True, status=status)
  print(jobs)
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

def add_zoom_inset(
  ax, zoom_region, inset_position=(0.6, 0.6, 0.3, 0.3),
  draw_rect=True, rect_kwargs=None, zoom_ax_kwargs=None
):
  """
  Adds a zoomed inset with full size and position control using inset_axes.

  Parameters:
  - ax: matplotlib.axes.Axes
      The main axes.
  - zoom_region: tuple (x1, x2, y1, y2)
      Limits of the region to zoom into.
  - inset_position: tuple (x0, y0, width, height)
      Inset position in axes fraction coordinates (not figure coords).
  - draw_rect: bool
      Draw a dashed rectangle on the main plot to show zoom region.
  - rect_kwargs: dict
      Styling for the zoom rectangle.
  - zoom_ax_kwargs: dict
      Dict of method calls on the inset axes.
  
  Returns:
  - axins: The inset axes object.
  """
  rect_kwargs = rect_kwargs or {'edgecolor': 'black', 'linestyle': 'dashed', 'linewidth': 1}
  zoom_ax_kwargs = zoom_ax_kwargs or {}

  # Create inset axes
  bbox = inset_position
  axins = inset_axes(
    ax,
    width="100%", height="100%",
    bbox_to_anchor=bbox,
    bbox_transform=ax.transAxes,
    loc='lower left',
    borderpad=0
  )

  # Set zoom limits
  x1, x2, y1, y2 = zoom_region
  axins.set_xlim(x1, x2)
  axins.set_ylim(y1, y2)

  # Copy each line fully
  for line in ax.get_lines():
    axins.plot(
      line.get_xdata(),
      line.get_ydata(),
      color=line.get_color(),
      linestyle=line.get_linestyle(),
      linewidth=line.get_linewidth(),
      marker=line.get_marker(),
      markersize=line.get_markersize(),
      markeredgecolor=line.get_markeredgecolor(),
      markerfacecolor=line.get_markerfacecolor(),
      alpha=line.get_alpha(),
      label=line.get_label(),
      zorder=line.get_zorder()
    )

  # Apply additional inset customizations
  for method, args in zoom_ax_kwargs.items():
    getattr(axins, method)(args) if isinstance(args, (list, tuple)) else getattr(axins, method)(args)

  # Draw rectangle on main plot
  if draw_rect:
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, **rect_kwargs)
    ax.add_patch(rect)

  return axins


def _plot(df: pd.DataFrame, cores: List[int] | None) -> None:
  marker_cycle = ["o", "s", "^", "d", "x", "P", "*", "v", ">"]
  color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
  hws_color_map = dict(zip(sorted(df['hardware'].unique()), itertools.cycle(color_cycle)))

  fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=False)
  axes = axes.flatten()

  for idx, func in enumerate(FUNCTIONS):
    ax = axes[idx]
    func_df = df[df["function"] == func]
    if cores:
      func_df = func_df[func_df['cores'].isin(cores)]

    func_df['bandwidth_GBps'] = func_df['bandwidth_MBps'] / 1e3

    for j, (hw, group) in enumerate(func_df.groupby("hardware", sort=False)):
      group_sorted = group.sort_values("cores")
      ax.plot(
        group_sorted["cores"],
        group_sorted["bandwidth_GBps"],
        color=hws_color_map.get(hw, hw),
        label=BOARD_NAMES_MAP.get(hw, hw),
        marker=marker_cycle[j % len(marker_cycle)],
        linewidth=1.8,
      )
    ax.set_xticks(list(func_df['cores'].unique()))
    ax.set_title(func, fontsize=FONT_TITLE)
    if idx >= 2: ax.set_xlabel("CPU cores")
    if idx % 2 == 0: ax.set_ylabel("Bandwidth [GB/s]")
    ax.grid(True, linestyle="-", alpha=0.8)
    ax.legend(loc='upper left')

    ## Add zoom
    zoom_cores_limit = 8
    max_y = func_df[func_df['cores'] <= zoom_cores_limit]['bandwidth_GBps'].max()
    min_y = func_df[func_df['cores'] <= zoom_cores_limit]['bandwidth_GBps'].min()
    zoom_ax = add_zoom_inset(
      ax,
      zoom_region=(0.0, 9.0, min_y*0.96, max_y*1.04),
      inset_position=(0.33, 0.1, 0.6, 0.72),  # x0, y0, width, height (ALL in percentage wrt ax size)
      rect_kwargs={'edgecolor': 'red', 'linestyle': '--', 'linewidth': 1},
      zoom_ax_kwargs={'grid': True, 'set_xticks': [2**p for p in range(8) if 2**p <= zoom_cores_limit]}
    )
    zoom_ax.yaxis.tick_right()
    for dir in ['top', 'right', 'bottom', 'left']:
      zoom_ax.spines[dir].set_linestyle(":")
      zoom_ax.spines[dir].set_edgecolor("red")

  # fig.suptitle("STREAM - Memory Bandwidth - Scaling", fontsize=17, y=0.97)
  fig.tight_layout() # (rect=[0, 0, 1, 0.95])

  ## Save plot
  path = OUT_DIR / f'STREAM.png'
  fig.savefig(path, dpi=300)
  print(f"[ OK ] figure saved to {path.resolve().absolute()}")

# ──────────────────────────────────────────────────────────────────────────────
# Unified CLI (sub-commands: files / sbm)
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_files(filepaths: List[Path | str]) -> pd.DataFrame:
  dfs = []
  for file in filepaths:
    cluster = Path(file).stem.split("_")[0]
    df = pd.read_csv(file)
    df["cluster"] = cluster
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)

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
  files_p.add_argument("-c", "--cores", nargs='+', help="A filter for the number of cores", default=None)

  # ── df sub-command ────────────────────────────────────────────────
  df_p = subparsers.add_parser("df", help="Parse a CSV input file")
  df_p.add_argument("inputs", nargs="+", type=Path, help="Input CSV text file(s)")
  df_p.add_argument("-c", "--cores", nargs='+', help="A filter for the number of cores", default=None)

  # ── sbm sub-command ────────────────────────────────────────────────
  sbm_p = subparsers.add_parser("sbm", help="Pull STREAM outputs from sbm jobs")
  sbm_p.add_argument("-s", "--status", nargs="+", default=["COMPLETED"], help="Job status filter")
  sbm_p.add_argument("-c", "--cores", nargs='+', help="A filter for the number of cores", default=None)

  args = parser.parse_args(argv)
  cores = [int(c) for c in args.cores] if args.cores else None

  if args.mode == "sbm":
    df = _build_dataframe_from_jobs(args.status)
    path = OUT_DIR / "STREAM_data.csv"
    df.to_csv(path, index=False)
    print(f"Wrote CSV summary with {len(df)} rows to {path.resolve().absolute()}")
  elif args.mode == "files":
    labels = _infer_hardware_labels(args.inputs, args.hardware)
    df = _build_dataframe_from_files(args.inputs, labels)
  elif args.mode == "df":
    df =load_csv_files(args.input)

  _plot(df, cores)


if __name__ == "__main__":
  main()

# Run example: python3 plots.py sbm -s COMPLETED -o results/stream.png