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
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.constants import *
from py_utils.utils import create_color_map, create_linestyle_map

plt.rc('axes', titlesize=FONT_AXES - 2)
plt.rc('axes', labelsize=FONT_AXES - 2)
plt.rc('xtick', labelsize=FONT_TICKS)
plt.rc('ytick', labelsize=FONT_TICKS)
plt.rc('legend', fontsize=FONT_LEGEND + 3)
plt.rc('figure', titlesize=FONT_TITLE)

# Configurable runtime columns
RUNTIME_BREAKDOWN_COLS = ["time_ddot", "time_mg", "time_opt", "time_spmv", "time_waxpby"]
DDOT_ALLREDUCE_COLS = ["ddot_allreduce_avg", "ddot_allreduce_max", "ddot_allreduce_min"]
SPMV_HALO_COLS = ["halo_avg", "halo_max", "halo_min"]


def load_data(csv_files):
  """Load multiple CSVs into a single DataFrame."""
  dfs = []
  for f in csv_files:
      df = pd.read_csv(f)
      df["source_file"] = os.path.basename(f)
      dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


def prepare_dataframe(df):
  """Prepare dataframe with proper types and column names."""
  # Ensure numeric types
  numeric_cols = ["processes", "threads", "gflops", "time_tot",
                  "gflops_ddot", "gflops_waxpby", "gflops_spmv", "gflops_mg"]
  numeric_cols.extend(RUNTIME_BREAKDOWN_COLS)
  numeric_cols.extend(DDOT_ALLREDUCE_COLS)
  numeric_cols.extend(SPMV_HALO_COLS)
  
  for col in numeric_cols:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  # Rename for plotting labels
  df = df.rename(columns={"processes": "nodes", "threads": "cpus_per_node"})
  
  return df


def plot_gflops_scaling(df, outdir="results"):
  """Plot GFLOP/s scaling across nodes."""
  plt.figure(figsize=(10, 7))
  cluster_color_map = create_color_map(df.sort_values('cluster')["cluster"].unique())

  for cluster, grp_cluster in df.groupby("cluster"):
    partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())
    
    for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
      grp_sorted = grp_cluster_partition.sort_values(["nodes",'partition'])
      plt.plot(
        grp_sorted["nodes"],
        grp_sorted["gflops"],
        marker="o",
        label=f"{cluster}-{partition}",
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
  print(f"Plot saved to {path.resolve().absolute()}")
  plt.close()


def plot_runtime_scaling(df, outdir="results"):
  """Plot runtime scaling across nodes."""
  plt.figure(figsize=(10, 7))
  cluster_color_map = create_color_map(df.sort_values('cluster')["cluster"].unique())

  for cluster, grp_cluster in df.groupby("cluster"):
    partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())
    
    for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
      grp_sorted = grp_cluster_partition.sort_values(["nodes",'partition'])
      plt.plot(
        grp_sorted["nodes"],
        grp_sorted["time_tot"],
        marker="o",
        label=f"{cluster}-{partition}",
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
  print(f"Plot saved to {path.resolve().absolute()}")
  plt.close()


def plot_gflops_breakdown_overall(df, outdir="results"):
  """Plot overall GFLOP/s breakdown (averaged across nodes)."""
  flop_cols = ["gflops_ddot", "gflops_waxpby", "gflops_spmv", "gflops_mg"]
  if not all(col in df.columns for col in flop_cols):
    print(f'[WARNING] Could not find all breakdown columns, skipping plot', flop_cols)
    return

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
  plt.title("HPCG GFLOPs Breakdown (Average)")
  plt.xticks(range(len(flop_cols)), [c.split('_')[1] for c in flop_cols])
  plt.xlabel("Kernel")
  plt.ylabel("GFLOPs")
  plt.legend(title="Cluster-Partition", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.tight_layout()
  plt.grid(True, axis='y', alpha=0.3)
  path = Path(outdir) / "HPCG_GFLOPs_Breakdown_Overall.png"
  plt.savefig(path, dpi=200)
  print(f"Plot saved to {path.resolve().absolute()}")
  plt.close()


def plot_gflops_breakdown_by_nodes(df, outdir="results"):
  """Plot GFLOP/s breakdown for each node count."""
  flop_cols = ["gflops_ddot", "gflops_waxpby", "gflops_spmv", "gflops_mg"]
  if not all(col in df.columns for col in flop_cols):
    print(f'[WARNING] Could not find all breakdown columns, skipping plot', flop_cols)
    return

  unique_nodes = sorted(df["nodes"].unique())
  
  for node_count in unique_nodes:
    # Create subfolder for this node count
    node_outdir = Path(outdir) / f"{node_count}_nodes"
    node_outdir.mkdir(parents=True, exist_ok=True)
    
    df_node = df[df["nodes"] == node_count].copy()
    
    # Create combined label
    df_node["cluster_partition"] = (
      df_node["cluster"] + "-" + df_node["partition"]
    )
    
    # Melt into long form
    df_melted = df_node.melt(
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
    plt.title(f"HPCG GFLOPs Breakdown ({node_count} Nodes)")
    plt.xticks(range(len(flop_cols)), [c.split('_')[1] for c in flop_cols])
    plt.xlabel("Kernel")
    plt.ylabel("GFLOPs")
    plt.legend(title="Cluster-Partition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.3)
    path = node_outdir / "HPCG_GFLOPs_Breakdown.png"
    plt.savefig(path, dpi=200)
    print(f"Plot saved to {path.resolve().absolute()}")
    plt.close()


def plot_runtime_breakdown_by_nodes(df, outdir="results"):
  """Plot runtime breakdown for each node count with detailed communication components."""
  available_cols = [col for col in RUNTIME_BREAKDOWN_COLS if col in df.columns]
  
  if not available_cols:
    print(f'[WARNING] No runtime breakdown columns found, skipping plot')
    return

  unique_nodes = sorted(df["nodes"].unique())
  
  for node_count in unique_nodes:
    # Create subfolder for this node count
    node_outdir = Path(outdir) / f"{node_count}_nodes"
    node_outdir.mkdir(parents=True, exist_ok=True)
    
    df_node = df[df["nodes"] == node_count].copy()
    df_node["cluster_partition"] = df_node["cluster"] + "-" + df_node["partition"]
    
    systems = sorted(df_node["cluster_partition"].unique())
    n_systems = len(systems)
    n_kernels = len(available_cols)
    
    # Create consistent color palette for systems
    system_colors = plt.cm.tab10(np.linspace(0, 1, n_systems))
    system_color_map = {sys: system_colors[i] for i, sys in enumerate(systems)}
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Setup bar positions
    x = np.arange(n_kernels)
    width = 0.8 / n_systems
    
    # Track legend entries
    legend_handles = []
    legend_labels = []
    
    for i, system in enumerate(systems):
      df_sys = df_node[df_node["cluster_partition"] == system]
      base_color = system_color_map[system]
      
      # Create darker version for communication (multiply RGB by 0.6)
      comm_color = tuple(list(base_color[:3] * 0.6) + [base_color[3]])
      
      for j, kernel_col in enumerate(available_cols):
        pos = x[j] + (i - n_systems/2 + 0.5) * width
        kernel_name = kernel_col.split('_')[1]
        
        total_time = df_sys[kernel_col].mean()
        
        # Check if this kernel has communication breakdown
        if kernel_name == "ddot" and all(col in df.columns for col in DDOT_ALLREDUCE_COLS):
          comm_avg = df_sys["ddot_allreduce_avg"].mean()
          comm_min = df_sys["ddot_allreduce_min"].mean()
          comm_max = df_sys["ddot_allreduce_max"].mean()
          compute_time = total_time - comm_avg
          
          # Plot compute (top) and communication (bottom)
          compute_bar = ax.bar(pos, compute_time, width, bottom=comm_avg, 
                              color=base_color, alpha=0.8)
          comm_bar = ax.bar(pos, comm_avg, width, color=comm_color, alpha=0.9)
          
          # Add error bar for communication showing min-max range
          yerr = [[comm_avg - comm_min], [comm_max - comm_avg]]
          ax.errorbar(pos, comm_avg, yerr=yerr, fmt='none', 
                     ecolor='black', capsize=3, capthick=1.5, alpha=0.6)
          
          if j == 0:  # Only add to legend once per system
            legend_handles.extend([compute_bar, comm_bar])
            legend_labels.extend([f"{system} (compute)", f"{system} (communication)"])
          
        elif kernel_name == "spmv" and all(col in df.columns for col in SPMV_HALO_COLS):
          comm_avg = df_sys["halo_avg"].mean()
          comm_min = df_sys["halo_min"].mean()
          comm_max = df_sys["halo_max"].mean()
          compute_time = total_time - comm_avg
          
          # Plot compute (top) and communication (bottom)
          compute_bar = ax.bar(pos, compute_time, width, bottom=comm_avg,
                              color=base_color, alpha=0.8)
          comm_bar = ax.bar(pos, comm_avg, width, color=comm_color, alpha=0.9)
          
          # Add error bar for communication showing min-max range
          yerr = [[comm_avg - comm_min], [comm_max - comm_avg]]
          ax.errorbar(pos, comm_avg, yerr=yerr, fmt='none',
                     ecolor='black', capsize=3, capthick=1.5, alpha=0.6)
          
          if j == 0:  # Only add to legend once per system
            legend_handles.extend([compute_bar, comm_bar])
            legend_labels.extend([f"{system} (compute)", f"{system} (communication)"])
        else:
          # Regular bar without breakdown
          bar = ax.bar(pos, total_time, width, color=base_color, alpha=0.8)
          
          if j == 0:  # Only add to legend once per system
            legend_handles.append(bar)
            legend_labels.append(system)
    
    ax.set_xlabel('Kernel')
    ax.set_ylabel('Time [s]')
    ax.set_title(f'HPCG Runtime Breakdown ({node_count} Nodes)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.split('_')[1] for c in available_cols])
    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    path = node_outdir / "HPCG_Runtime_Breakdown.png"
    plt.savefig(path, dpi=200)
    print(f"Plot saved to {path.resolve().absolute()}")
    plt.close()


def plot_ddot_detailed_breakdown(df, outdir="results"):
  """Plot detailed ddot breakdown showing allreduce components."""
  # This function is deprecated - communication details are now shown in plot_runtime_breakdown_by_nodes
  pass


def plot_spmv_detailed_breakdown(df, outdir="results"):
  """Plot detailed spmv breakdown showing halo exchange components."""
  # This function is deprecated - communication details are now shown in plot_runtime_breakdown_by_nodes
  pass


def main():
  parser = argparse.ArgumentParser(description="Generate comparison scaling plots from HPCG CSVs")
  parser.add_argument("csv_files", nargs="+", help="Input CSV files (from parse_hpcg.py)")
  parser.add_argument("--outdir", default="results", help="Output directory for plots")
  args = parser.parse_args()

  # Create output directory
  Path(args.outdir).mkdir(parents=True, exist_ok=True)

  # Load and prepare data
  df = load_data(args.csv_files)

  required = {"cluster", "partition", "processes", "threads", "gflops", "time_tot"}
  if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {', '.join(required)}")

  # Map names and prepare dataframe
  df['cluster'] = df['cluster'].map(CLUSTER_NAMES_MAP)
  df['partition'] = df['partition'].map(PARTITION_NAMES_MAP)
  df = prepare_dataframe(df)
  
  print("Loaded data:")
  print(df)
  print("\nGenerating plots...")

  # Generate all plots
  plot_gflops_scaling(df, args.outdir)
  plot_runtime_scaling(df, args.outdir)
  plot_gflops_breakdown_overall(df, args.outdir)
  plot_gflops_breakdown_by_nodes(df, args.outdir)
  plot_runtime_breakdown_by_nodes(df, args.outdir)
  plot_ddot_detailed_breakdown(df, args.outdir)
  plot_spmv_detailed_breakdown(df, args.outdir)
  
  print("\nAll plots generated successfully!")


if __name__ == "__main__":
  main()