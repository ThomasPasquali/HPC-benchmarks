#!/usr/bin/env python3
"""
plots.py

Usage:
  python plots.py results1.csv results2.csv ... [--outdir plots]

Generates scaling plots comparing clusters and partitions.
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.constants import *

FONT_LEGEND += 4
plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

def load_data(csv_files):
  """Load multiple CSVs into a single DataFrame."""
  dfs = []
  for f in csv_files:
      df = pd.read_csv(f)
      df["source_file"] = os.path.basename(f)
      dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


def plot_scaling(df: pd.DataFrame, outdir="plots"):
  os.makedirs(outdir, exist_ok=True)

  # Ensure numeric types
  for col in ["processes", "threads", "gflops", "time_sec"]:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  # Rename for plotting labels
  df = df.rename(columns={"processes": "nodes", "threads": "cpus_per_node"})

  # ---- GFLOP/s comparison ----
  plt.figure(figsize=(10, 7))
  for (cluster, partition, cpus), subdf in df.groupby(["cluster", "partition", "cpus_per_node"]):
    subdf = subdf.sort_values("nodes")
    label = f"{cluster} / {partition} - {cpus} CPUs/node"
    plt.plot(subdf["nodes"], subdf["gflops"], marker="o", label=label)

  plt.xticks(df["nodes"].unique())
  plt.xlabel("Nodes")
  plt.ylabel("GFLOP/s")
  plt.title("HPCG - GFLOP/s")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.tight_layout()
  plt.savefig(os.path.join(outdir, "comparison_scaling_gflops.png"))
  plt.close()

  # ---- Time comparison ----
  plt.figure(figsize=(10, 7))
  for (cluster, partition, cpus), subdf in df.groupby(["cluster", "partition", "cpus_per_node"]):
      subdf = subdf.sort_values("nodes")
      label = f"{cluster} / {partition} - {cpus} CPUs/node"
      plt.plot(subdf["nodes"], subdf["time_sec"], marker="o", label=label)

  plt.xticks(df["nodes"].unique())
  plt.xlabel("Nodes")
  plt.ylabel("Runtime [s]")
  plt.title("HPCG - Time")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.tight_layout()
  plt.savefig(os.path.join(outdir, "comparison_scaling_time.png"))
  plt.close()

  print(f"Comparison plots saved in {outdir}/")


def main():
  parser = argparse.ArgumentParser(description="Generate comparison scaling plots from HPCG CSVs")
  parser.add_argument("csv_files", nargs="+", help="Input CSV files (from parse_hpcg.py)")
  parser.add_argument("--outdir", "-o", default="plots", help="Output directory for plots")
  args = parser.parse_args()

  df = load_data(args.csv_files)

  required = {"cluster", "partition", "processes", "threads", "gflops", "time_sec"}
  if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {', '.join(required)}")

  plot_scaling(df, args.outdir)


if __name__ == "__main__":
  main()