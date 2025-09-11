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
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.constants import *
from py_utils.utils import create_color_map, create_linestyle_map

plt.rc('axes', titlesize=FONT_AXES - 2)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES - 2)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND + 3)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

def load_data(csv_files):
  """Load multiple CSVs into a single DataFrame."""
  dfs = []
  for f in csv_files:
      df = pd.read_csv(f)
      df["source_file"] = os.path.basename(f)
      dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


def plot_scaling(df: pd.DataFrame, outdir="results"):
  # Ensure numeric types
  for col in ["processes", "threads", "gflops", "time_sec",
              "gflops_ddot", "gflops_waxpby", "gflops_spmv", "gflops_mg"]:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  # Rename for plotting labels
  df = df.rename(columns={"processes": "nodes", "threads": "cpus_per_node"})

  # ---- GFLOP/s comparison ----
  plt.figure(figsize=(10, 7))
  ## for (cluster, partition, cpus), subdf in df.groupby(["cluster", "partition", "cpus_per_node"]):
  cluster_color_map = create_color_map(df.sort_values('cluster')["cluster"].unique())

  for cluster, grp_cluster in df.groupby("cluster"):
    partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())
    
    for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
      grp_sorted = grp_cluster_partition.sort_values(["nodes",'partition'])
      plt.plot(
        grp_sorted["nodes"],
        grp_sorted["gflops"],
        marker="o",
        label=f"{cluster}-{partition}", # - {cpus} CPUs/node",
        color=cluster_color_map[cluster],
        linestyle=partition_linestyles[partition],
      )

  plt.xticks(df["nodes"].unique())
  plt.xlabel("Nodes")
  plt.ylabel("GFLOPs")
  plt.title("HPCG Scaling")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.tight_layout()
  path = Path(outdir) / "HPCG_Scaling_GFLOPs.png"
  plt.savefig(path, dpi=200)
  print(f"Plot saved to {path.resolve().absolute()}/")
  plt.close()

  # ---- Time comparison ----
  plt.figure(figsize=(10, 7))
  for cluster, grp_cluster in df.groupby("cluster"):
    partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())
    
    for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
      grp_sorted = grp_cluster_partition.sort_values(["nodes",'partition'])
      plt.plot(
        grp_sorted["nodes"],
        grp_sorted["time_sec"],
        marker="o",
        label=f"{cluster}-{partition}", # - {cpus} CPUs/node",
        color=cluster_color_map[cluster],
        linestyle=partition_linestyles[partition],
      )
      
  plt.xticks(df["nodes"].unique())
  plt.xlabel("Nodes")
  plt.ylabel("Runtime [s]")
  plt.title("HPCG Scaling")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.tight_layout()
  path = Path(outdir) / "HPCG_Scaling_Runtime.png"
  plt.savefig(path, dpi=200)
  print(f"Plot saved to {path.resolve().absolute()}/")
  plt.close()
  
  # ---- FLOP Breakdown Comparison ----
  flop_cols = ["gflops_ddot", "gflops_waxpby", "gflops_spmv", "gflops_mg"]
  if all(col in df.columns for col in flop_cols):
    # Aggregate over nodes/cpus (mean per cluster-partition)
    df_breakdown = (
      df.groupby(["cluster", "partition"])[flop_cols]
        .mean()
        .reset_index()
    )

    # Create combined label for hue
    df_breakdown["cluster_partition"] = (
      df_breakdown["cluster"] + "-" + df_breakdown["partition"]
    )

    # Melt into long form for seaborn
    df_melted = df_breakdown.melt(
      id_vars=["cluster_partition"],
      value_vars=flop_cols,
      var_name="kernel",
      value_name="GFLOPs"
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
      data=df_melted,
      x="kernel",
      y="GFLOPs",
      hue="cluster_partition",
      errorbar=None
    )
    plt.title("HPCG FLOPs Breakdown")
    plt.xticks(range(len(flop_cols)), [c.split('_')[1] for c in flop_cols])
    plt.xlabel("Kernel")
    plt.ylabel("GFLOPs")
    plt.legend(title="Cluster-Partition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True)
    path = Path(outdir) / "HPCG_FLOP_Breakdown.png"
    plt.savefig(path, dpi=200)
    print(f"Plot saved to {path.resolve().absolute()}/")
    plt.close()
  else:
    print(f'[WARNING] Could not find all breakdown columns, skipping plot', flop_cols)


def main():
  parser = argparse.ArgumentParser(description="Generate comparison scaling plots from HPCG CSVs")
  parser.add_argument("csv_files", nargs="+", help="Input CSV files (from parse_hpcg.py)")
  args = parser.parse_args()

  df = load_data(args.csv_files)

  required = {"cluster", "partition", "processes", "threads", "gflops", "time_sec"}
  if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {', '.join(required)}")

  # Map names
  df['cluster'] = df['cluster'].map(CLUSTER_NAMES_MAP)
  df['partition'] = df['partition'].map(PARTITION_NAMES_MAP)
  print(df)

  plot_scaling(df)


if __name__ == "__main__":
  main()