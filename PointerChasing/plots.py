#!/usr/bin/env python3
"""
Generate the pointer-chasing benchmark plots originally produced with gnuplot.

• random-chase.png - avg access time vs. memory area (log-scaled X)   with L1/L2/L3 cache size markers.

• linear-chase.png - avg access time vs. stride size.

• fused-linear-chase.png - data-access bandwidth vs. stride for fuse factors 1-8.
"""

from pathlib import Path
import argparse
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import sbatchman as sbm

# TODO
HW_SPECS = {
  'pioneer':None,
  'bananaf3':None,
  'arriesgado':None,
}


def plot_random_chase(src: Path, dst: Path, hw_name: str) -> None:
  data = np.genfromtxt(src, skip_header=2, usecols=(0, 1), unpack=True)
  x, y = data[0], data[1]

  fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
  ax.plot(x, y, marker="o", markersize=3, linewidth=1, label=hw_name)

  # log-scale on X
  ax.set_xscale("log")
  ax.set_xlabel("memory area in bytes")
  ax.set_ylabel("avg access time in ns")
  ax.set_title("Access times in dependence of memory area")

  # draw vertical cache-size markers (blue)
  l1, l2, l3 = 32 * 1024, 256 * 1024, 12_288 * 1024  # bytes
  y_max = y.max()
  for pos in (l1, l2, l3):
      ax.axvline(x=pos, ymin=0, ymax=1, linewidth=1)

  ax.set_ylim(0, y_max * 1.05)
  ax.legend()
  fig.tight_layout()
  fig.savefig(dst)
  print(f'Plot "random-chase" saved to {dst.resolve().absolute()}')
  plt.close(fig)


def plot_linear_chase(src: Path, dst: Path, hw_name: str) -> None:
  data = np.genfromtxt(src, skip_header=2, usecols=(0, 1), unpack=True)
  x, y = data[0], data[1]

  fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
  ax.plot(x, y, marker="o", markersize=3, linewidth=1, label=hw_name)

  ax.set_xlabel("stride in bytes")
  ax.set_ylabel("avg access time in ns")
  ax.set_title("Access times in dependence of stride")
  ax.legend()
  fig.tight_layout()
  fig.savefig(dst)
  print(f'Plot "linear-chase" saved to {dst.resolve().absolute()}')
  plt.close(fig)


def plot_fused_linear_chase(src: Path, dst: Path, hw_name: str) -> None:
  try:
    raw = np.genfromtxt(src, skip_header=4)
  except ValueError:
    raw = np.genfromtxt(src, skip_header=4, skip_footer=1)
  x = raw[:, 0]
  print(x)

  fig, ax = plt.subplots(figsize=(15, 9), dpi=100)
  for fuse in range(8):  # columns 2-9 = fuse 1-8
    y = raw[:, fuse + 1]
    ax.plot(x, y, marker="o", markersize=3, linewidth=1, label=f"fuse {fuse+1}")

  ax.set_xlabel("stride in bytes")
  ax.set_ylabel("data access speed in GiB/s")
  ax.set_title(f"Data access speeds in dependence of stride and fuse {hw_name}")
  ax.legend()
  fig.tight_layout()
  fig.savefig(dst)
  print(f'Plot "fused-linear-chase" saved to {dst.resolve().absolute()}')
  plt.close(fig)

# =====================================================

def filter_jobs(jobs: List[sbm.Job]) -> List[sbm.Job]:
  filtered_jobs = []
  for job in jobs:
    if job.status in ['COMPLETED']:
      filtered_jobs.append(job)
  return filtered_jobs

# =====================================================

def main() -> None:
  parser = argparse.ArgumentParser(description="Generate pointer-chase plots.")
  parser.add_argument("--output-dir", default=Path("./results"), type=Path, help="where PNGs are written")
  args = parser.parse_args()

  outdir: Path = args.output_dir
  outdir.mkdir(parents=True, exist_ok=True)

  # jobs = filter_jobs(sbm.jobs_list(from_active=True, from_archived=True))
  jobs = sbm.jobs_list(from_active=False, from_archived=True)
  plot_functions = {
    "linear-chase": plot_linear_chase,
    "random-chase": plot_random_chase,
    "fused-linear-chase" : plot_fused_linear_chase,
  }
  done_plots = {
    "linear-chase": False,
    "random-chase": False,
    "fused-linear-chase" : False,
  }
  for job in jobs:
    program = job.tag

    if program not in plot_functions.keys():
      print(f'Unrecognized program "{program}, skipping"')
      continue

    if not done_plots[program]:
      print('\n\n')
      print('='*50)
      print(job.get_stdout())
      hw_name = job.config_name
      plot_functions[program](job.get_stdout_path(), outdir / f'{hw_name}_{program}.png', hw_name)
    else:
      print(f'Plot for {program} already generated from another job, skipping')
      continue

  print("✔ Plots written to", outdir.resolve().absolute())


if __name__ == "__main__":
  main()
