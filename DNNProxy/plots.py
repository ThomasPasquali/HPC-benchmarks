from csv import Error
import itertools
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.constants import *
from py_utils.utils import create_color_map

FONT_TITLE = 18
FONT_AXES = 18
FONT_TICKS = 16
FONT_LEGEND = 12

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the model names before the mapping
MODELS_BLACKLIST = [] # ['ResNet-152']
MODEL_NAME_MAP = {
  'DLRM': 'DLRM',
  'ResNet-152': 'ResNet-152',
  'bert': 'BERT',
  'gpt2': 'GPT2',
  'ResNet-50-allreduce': 'ResNet-50-AllRed',
  'ResNet-50-ring': 'ResNet-50-Ring',
}
MODEL_IS_COMM_ONLY = {
  'DLRM':                 False,
  'ResNet-152':           False,
  'BERT':                 True,
  'GPT2':                 True,
  'ResNet-50-AllRed':     True,
  'ResNet-50-Ring':       True,
}

def plot_scaling_by_model(
  df: pd.DataFrame,
  clusters_partitions: Union[Tuple[str, str], List[Tuple[str, str]]],
  *,
  model_color_map: Union[None, Dict[str, str]] = None,
  node_col: str = "nodes",
  time_col: str = "geomean_time",
  time_std_col: str = "std_time",
  model_col: str = "model",
  figsize: tuple = (8, 12),
  marker: str = "o"
) -> plt.Axes:
  """
  Plot scaling curves (time vs. nodes) for each model on a single figure,
  optionally comparing multiple (cluster, partition) combinations.

  Parameters
  ----------
  df : pd.DataFrame
      Full performance DataFrame.
  clusters_partitions : tuple or list of tuples
      One or more (cluster, partition) combinations to plot.
  model_color_map : dict, optional
      Optional map from model name to color.
  node_col, time_col, model_col : str
      Column names for x-axis, y-axis, and model grouping.
  figsize : tuple
      Figure size.
  marker : str
      Marker style.

  Returns
  -------
  matplotlib.axes.Axes
      The axis with the plot.
  """

  # Normalize input to a list of (cluster, partition) pairs
  if isinstance(clusters_partitions, tuple):
    clusters_partitions = [clusters_partitions]

  # Generate unique line styles (cycled)
  line_styles = itertools.cycle(["-", "--", "-.", ":"])

  fig, ax = plt.subplots(figsize=figsize)

  for (cluster, partition) in clusters_partitions:
    mask = (df["cluster"] == cluster) & (df["partition"] == partition)
    slice_df = df.loc[mask].copy()

    if slice_df.empty:
      raise ValueError(f"No data for cluster='{cluster}' & partition='{partition}'")

    slice_df.sort_values(node_col, inplace=True)
    style = next(line_styles)

    # Build a global model list so offsets are consistent across all groups
    model_list = list(slice_df[model_col].unique())
    n_models = len(model_list)

    for model, grp in slice_df.groupby(model_col):
      label = f"{model} ({cluster}-{partition})"
      
      # Global offset index
      # model_idx = model_list.index(model)
      # Spread offsets evenly around 0
      offset = 0 # (model_idx - (n_models - 1) / 2) * 0.15 if n_models > 1 else 0

      # Apply offset to x values
      x_vals = grp[node_col] + offset

      ax.errorbar(
        x_vals,
        grp[time_col],
        yerr=grp[time_std_col],
        label=label,
        marker=marker,
        linestyle=style,
        color=model_color_map[str(model)] if model_color_map else None,
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        ecolor=model_color_map[str(model)] if model_color_map else None,
      )

  all_xticks = sorted(df[node_col].unique())
  ax.set_xticks(all_xticks)
  ax.set_xlabel("Number of Nodes")
  ax.set_ylabel("Geometric Mean Time [s]")

  if len(clusters_partitions) == 1:
    cluster, partition = clusters_partitions[0]
    ax.set_title(f"Strong Scaling - Cluster={cluster} - Partition={partition}")
  else:
    ax.set_title("Strong Scaling Comparison Across Clusters/Partitions")

  ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
  ax.legend(title="Model (Cluster-Partition/Configuration)")
  fig.tight_layout()

  return ax


def plot_performance_ratio(
  df: pd.DataFrame,
  ref_cluster: str,
  ref_partition: str,
  cmp_cluster: str,
  cmp_partition: str,
  *,
  node_col: str = "nodes",
  time_col: str = "geomean_time",
  model_col: str = "model",
  figsize: tuple = (8, 12),
  marker: str = "o",
  is_barplot=False,
) -> plt.Axes:
  """
  Plot performance ratio (ref_time / cmp_time) for each model over node counts.

  Parameters
  ----------
  df : pd.DataFrame
      The full performance DataFrame.
  ref_cluster, ref_partition : str
      The reference cluster and partition.
  cmp_cluster, cmp_partition : str
      The cluster and partition to compare against.
  node_col, time_col, model_col : str
      Column names for x-axis, y-axis, and model.
  figsize : tuple
      Size of the matplotlib figure.
  marker : str
      Marker for the line plot.

  Returns
  -------
  matplotlib.axes.Axes
      The axis with the plot.
  """
  # Filter the two slices
  ref_df = df[(df["cluster"] == ref_cluster) & (df["partition"] == ref_partition)]
  cmp_df = df[(df["cluster"] == cmp_cluster) & (df["partition"] == cmp_partition)]

  if ref_df.empty or cmp_df.empty:
    raise ValueError("One of the cluster/partition combinations has no data.")

  # Merge on (nodes, model) to align comparable measurements
  merged = pd.merge(
    ref_df[[node_col, model_col, time_col]],
    cmp_df[[node_col, model_col, time_col]],
    on=[node_col, model_col],
    suffixes=('_ref', '_cmp')
  )

  # Compute ratio
  merged["ratio"] = merged[f"{time_col}_ref"] / merged[f"{time_col}_cmp"]

  fig, ax = plt.subplots(figsize=figsize)

  # Plot each model separately
  if is_barplot:
    models = merged[model_col].unique()
    num_models = len(models)
    bar_width = 0.8 / num_models  # Total width for all bars per group is 0.8

    node_vals = sorted(merged[node_col].unique())
    node_pos = np.arange(len(node_vals))  # Use positions as x-axis for bar alignment

    for i, model in enumerate(models):
      grp = merged[merged[model_col] == model]
      # Align grp data with node positions
      grp = grp.set_index(node_col).reindex(node_vals).reset_index()

      ax.bar(
        node_pos + i * bar_width,
        grp["ratio"],
        width=bar_width,
        label=model,
      )

    ax.set_xticks(node_pos + bar_width * (num_models - 1) / 2)
    ax.set_xticklabels(node_vals)
  else:
    for model, grp in merged.groupby(model_col):
      ax.plot(
        grp[node_col],
        grp["ratio"],
        label=model,
        marker=marker
      )

    xticks = sorted(merged[node_col].unique())
    ax.set_xticks(xticks)
    
  ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Parity')
  ax.set_xlabel("Number of Nodes")
  ax.set_ylabel(f"Runtime Ratio ({ref_cluster}-{ref_partition} / {cmp_cluster}-{cmp_partition})")
  ax.set_title(f"Runtime Ratio: {ref_cluster}-{ref_partition} vs. {cmp_cluster}-{cmp_partition}")
  ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
  ax.legend(title="Model")
  fig.tight_layout()

  return ax

PLOTS_COMPARISON_SHOW_STD = False
PLOTS_SPLIT_COMM_ONLY_AND_EMULATED_COMPUTE = True
def plot_barplot_comparisons(df: pd.DataFrame, min_coverage_ratio=0.5):
  df['cluster-partition'] = df['cluster'] + '-' + df['partition']
  cluster_partitions = df['cluster-partition'].unique()
  all_models = sorted(df["model"].unique())

  # Split models into two groups if enabled
  if PLOTS_SPLIT_COMM_ONLY_AND_EMULATED_COMPUTE:
    comm_only_models = [m for m in all_models if MODEL_IS_COMM_ONLY.get(m, False)]
    emu_compute_models = [m for m in all_models if not MODEL_IS_COMM_ONLY.get(m, False)]
    model_groups = {
      "comm_only": comm_only_models,
      "emulated_compute": emu_compute_models
    }
  else:
    model_groups = {"all": all_models}

  for group_name, models in model_groups.items():
    if not models:
        continue

    # Assign unique colors to implementations (group-specific)
    model_colors = create_color_map(models)

    for cp1, cp2 in itertools.combinations(cluster_partitions, 2):
        fig, ax = plt.subplots(figsize=(12, 5))
        bar_width = 0.8 / len(models)
        miny, maxy = np.inf, -np.inf

        # --- Build cpu_data: {nodes: {model: speedup, std_cp1, std_cp2}} ---
        cpu_data = {}
        for model in models:
            b1_df = df[(df["model"] == model) & (df["cluster-partition"] == cp1)]
            b2_df = df[(df["model"] == model) & (df["cluster-partition"] == cp2)]
            merged = pd.merge(
                b1_df, b2_df,
                on="nodes",
                suffixes=(f"_{cp1}", f"_{cp2}")
            )
            for _, row in merged.iterrows():
                cpu = int(row["nodes"])
                if cpu not in cpu_data:
                    cpu_data[cpu] = {}
                r1 = row[f"geomean_time_{cp1}"]
                r2 = row[f"geomean_time_{cp2}"]
                spd = np.where(r1 < r2, -(r2 / r1) + 1, (r1 / r2) - 1)
                cpu_data[cpu][model] = {
                    "speedup": spd,
                    "std_cp1": row.get(f"std_time_{cp1}", 0.0),
                    "std_cp2": row.get(f"std_time_{cp2}", 0.0),
                }

        # filter nodes with enough coverage
        valid_cpu_counts = [
            cpu for cpu, models_dict in cpu_data.items()
            if len(models_dict) / len(models) >= min_coverage_ratio
        ]
        if not valid_cpu_counts:
            plt.close()
            continue

        valid_cpu_counts = sorted(valid_cpu_counts)
        x_base = np.arange(len(valid_cpu_counts))
        offsets = np.linspace(-0.4 + bar_width / 2,
                              0.4 - bar_width / 2,
                              len(models))

        # --- Plot per model ---
        for i, model in enumerate(models):
            speedups, x_pos, stds_cp1, stds_cp2 = [], [], [], []
            for idx, cpu in enumerate(valid_cpu_counts):
                if model in cpu_data[cpu]:
                    entry = cpu_data[cpu][model]
                    speedups.append(entry["speedup"])
                    stds_cp1.append(entry["std_cp1"])
                    stds_cp2.append(entry["std_cp2"])
                    x_pos.append(x_base[idx] + offsets[i])

            if not speedups:
                continue

            bars = ax.bar(
                x_pos, speedups,
                width=bar_width,
                color=model_colors[model],
                label=model
            )
            ax.axhline(0, color='r', linewidth=1)

            for bar, spd, s1, s2 in zip(bars, speedups, stds_cp1, stds_cp2):
                x_center = bar.get_x() + bar.get_width() / 2
                height = bar.get_height()
                dx = bar_width * 0.25

                if PLOTS_COMPARISON_SHOW_STD:
                    # std indicator for cp1
                    ax.plot([x_center - dx, x_center - dx],
                            [height, height + s1],
                            color="black", linewidth=1)
                    ax.hlines(height + s1,
                              x_center - dx - 0.05, x_center - dx + 0.05,
                              color="black")

                    # std indicator for cp2
                    ax.plot([x_center + dx, x_center + dx],
                            [height, height + s2],
                            color="red", linewidth=1)
                    ax.hlines(height + s2,
                              x_center + dx - 0.05, x_center + dx + 0.05,
                              color="red")

                if PLOTS_COMPARISON_SHOW_STD:
                  ymax = height + max(s1, s2)
                  miny = min(miny, height - max(s1, s2))
                else:
                  ymax = height
                  miny = min(miny, height)
                  
                maxy = max(maxy, ymax)
                  
                text_offset = min(0.05, 0.05 * abs(ymax) + 0.02)  # relative offset (scales with bar height, with a small constant)

                if ymax >= 0:
                  y_text = ymax + text_offset
                  va = "bottom"
                else:
                  y_text = ymax - text_offset
                  va = "top"

                ax.text(
                  x_center, y_text,
                  f"{abs(spd)+1:.2f}x",
                  ha="center", va=va, fontsize=10
                )

        # --- Axis/labels formatting ---
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(cpu) for cpu in valid_cpu_counts])
        yticks = np.arange(-np.ceil(np.abs(miny)), np.ceil(maxy), 1)
        ax.set_yticks(yticks)
        yticklabels = []
        yticklabels_start = yticks[0]-1
        for i in range(len(yticks)):
            tick = yticklabels_start+i
            if tick >= -1:
                tick += 2
            yticklabels.append(f'${int(abs(tick))}\\times$')
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Speedup")
        if SET_FIG_TITLE:
            ax.set_title(f"{cp1} vs {cp2} [{group_name}]")
        miny = yticks[0]-1.0
        maxy = yticks[-1]+1.0
        ax.set_ylim(miny, maxy)
        ax.text(-0.55, maxy-0.1,  f'{cp2} Faster',
                fontsize=18, ha='left', va='top')
        ax.text(-0.55, miny+0.05, f'{cp1} Faster',
                fontsize=18, ha='left', va='bottom')
        # Place legend above the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=len(models))
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        fig.tight_layout()

        out_path = OUT_DIR / 'comparison' / f'DNNProxy_compare_{cp1}_vs_{cp2}_{group_name}.png'
        out_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_path)
        plt.close()
        print(f"Grouped comparison plot saved to {out_path.resolve()}")


def main():
  data_paths = [Path(p) for p in sys.argv[1:] if Path(p).exists() and Path(p).is_file()]
  if not data_paths:
    print("Please pass a list of CSV files as argument")
    exit(1)
    
  print(f'Reading data from files: {[str(p) for p in data_paths]}')
  dfs = [pd.read_csv(p) for p in data_paths]
  df = pd.concat(dfs, ignore_index=True)
  # Map names
  df['cluster'] = df['cluster'].map(CLUSTER_NAMES_MAP)
  df['partition'] = df['partition'].map(PARTITION_NAMES_MAP)
  df['model'] = df['model'].map(MODEL_NAME_MAP)
  # Filter
  if MODELS_BLACKLIST:
    df = df[~df['model'].isin(MODELS_BLACKLIST)]

  print(df)

  models = df['model'].unique()
  model_color_map = create_color_map(models)
  
  plot_barplot_comparisons(df)

  for cluster, partition in df.groupby(["cluster", "partition"]).groups.keys():
    plot_scaling_by_model(df, clusters_partitions=(cluster, partition), model_color_map=model_color_map)
    filename = f"DNNProxy_scaling_{cluster}-{partition}"
    path = OUT_DIR / "scaling_individual" / filename
    path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Plot saved to {path.resolve().absolute()}")

  combos = df[["cluster", "partition"]].drop_duplicates()
  pairs = [(a, b) for a in combos.itertuples(index=False, name=None) for b in combos.itertuples(index=False, name=None) if a != b]
  for (ref_cluster, ref_partition), (cmp_cluster, cmp_partition) in pairs:
    try:
      plot_performance_ratio(
        df,
        ref_cluster, ref_partition,
        cmp_cluster, cmp_partition
      )
      filename = f"DNNProxy_ratio_line_{ref_cluster}-{ref_partition}_vs_{cmp_cluster}-{cmp_partition}.png"
      path = OUT_DIR / "ratio_line" / filename
      path.parent.mkdir(exist_ok=True, parents=True)
      plt.savefig(path, dpi=200)
      plt.close()
      print(f"Plot saved to {path.resolve().absolute()}")

      plot_performance_ratio(
        df,
        ref_cluster, ref_partition,
        cmp_cluster, cmp_partition,
        is_barplot=True,
      )
      filename = f"DNNProxy_ratio_boxplot_{ref_cluster}-{ref_partition}_vs_{cmp_cluster}-{cmp_partition}.png"
      path = OUT_DIR / "ratio_boxplot" / filename
      path.parent.mkdir(exist_ok=True, parents=True)
      plt.savefig(path, dpi=200)
      plt.close()
      print(f"Plot saved to {path.resolve().absolute()}")
    except ValueError as e:
      print(f"Skipping {ref_cluster}/{ref_partition} vs {cmp_cluster}/{cmp_partition}: {e}")


  pairs = list(itertools.combinations(combos.itertuples(index=False, name=None), 2))
  for (ref_cluster, ref_partition), (cmp_cluster, cmp_partition) in pairs:
    try:
      plot_scaling_by_model(df, clusters_partitions=[(ref_cluster, ref_partition), (cmp_cluster, cmp_partition)], model_color_map=model_color_map)
      filename = f"DNNProxy_scaling_comp_{ref_cluster}-{ref_partition}_vs_{cmp_cluster}-{cmp_partition}.png"
      path = OUT_DIR / "scaling_comp" / filename
      path.parent.mkdir(exist_ok=True, parents=True)
      plt.savefig(path, dpi=200)
      plt.close()
      print(f"Plot saved to {path.resolve().absolute()}")
    except ValueError as e:
      print(f"Skipping {ref_cluster}/{ref_partition} vs {cmp_cluster}/{cmp_partition}: {e}")


if __name__ == "__main__":
  main()
