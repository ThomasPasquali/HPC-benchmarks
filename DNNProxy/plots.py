from csv import Error
import itertools
from pathlib import Path
from pprint import pprint
from statistics import geometric_mean, stdev
import sys
import sbatchman as sbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple
import re

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

def plot_scaling_by_model(
  df: pd.DataFrame,
  clusters_partitions: Union[Tuple[str, str], List[Tuple[str, str]]],
  *,
  model_color_map: Union[None, Dict[str, str]] = None,
  node_col: str = "nodes",
  time_col: str = "geomean_time",
  time_std_col: str = "std_time",
  model_col: str = "model",
  figsize: tuple = (10, 6),
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

    # Plot each model with a distinct color, and line style per (cluster, partition)
    for model, grp in slice_df.groupby(model_col):
      label = f"{model} ({cluster}-{partition})"
      ax.errorbar(
        grp[node_col],
        grp[time_col],
        yerr=grp[time_std_col],
        label=label,
        marker=marker,
        linestyle=style,
        color=model_color_map[str(model)] if model_color_map else None
      )

  all_xticks = sorted(df[node_col].unique())
  ax.set_xticks(all_xticks)
  ax.set_xlabel("Number of Nodes")
  ax.set_ylabel("Geometric Mean Time (s)")

  if len(clusters_partitions) == 1:
    cluster, partition = clusters_partitions[0]
    ax.set_title(f"Strong Scaling - Cluster={cluster} - Partition={partition}")
  else:
    ax.set_title("Strong Scaling Comparison Across Clusters/Partitions")

  ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
  ax.legend(title="Model (Cluster-Partition)")
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
  figsize: tuple = (10, 6),
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
  ax.set_ylabel(f"Performance Ratio ({ref_cluster}-{ref_partition} / {cmp_cluster}-{cmp_partition})")
  ax.set_title(f"Performance Ratio: {ref_cluster}-{ref_partition} vs. {cmp_cluster}-{cmp_partition}")
  ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
  ax.legend(title="Model")
  fig.tight_layout()

  return ax


def parse_stdout(job: sbm.Job) -> Dict[str, List[float]]:
  stdout = job.get_stdout()
  if stdout is None:
    raise Exception(f'Job stdout empty:\n{job}')
  
  model = None
  if 'GPT2-' in stdout:
    model = 'gpt2'
  elif 'Bert-' in stdout:
    model = 'bert'
  elif 'ResNet-152' in stdout:
    model = 'ResNet-152'
  elif 'DLRM ' in stdout:
    model = 'DLRM'
  elif 'ResNet-50' in stdout:
    if '(allreduce)' in stdout:
      model = 'resnet-allreduce'
    else:
      model = 'resnet-ring'
      
  if model is None:
    raise Error(f'Could not find the model in output of job: {job}\n{stdout}')
  
  lines = stdout.splitlines()
  times = {}

  if model in ['DLRM', 'ResNet-152']:
    lines = [lines[-1]]
    
  for line in lines:
    parts = line.split(', ') if ', ' in line else [line]
    _, time = parts[-1].split(' = ')
    time = float(time.split(' ')[0])

    if model not in times: times[model] = []
    times[model].append(time)

  return times

def filter_jobs(jobs: List[sbm.Job]) -> List[sbm.Job]:
  filtered_jobs = []
  for job in jobs:
    if job.status in ['COMPLETED']:
      filtered_jobs.append(job)
  return filtered_jobs


def main():
  data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
  if data_path and data_path.exists() and data_path.is_file():
    print(f'Reading data from file: "{data_path}"')
    df = pd.read_csv(data_path)
  else:
    jobs = filter_jobs(sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=True))
    data = []

    for job in jobs:
      # print('='*50)
      # pprint(job)
      # print(job.get_stdout())
      # print('-'*50)
      
      res = parse_stdout(job)
      # print(res)
      for model, times in res.items():
        m = re.match(r'(\w+)_(\d+)nodes', job.config_name)
        print(model)
        print(times)
        data.append({
          'cluster': job.cluster_name,
          'partition': m.group(1),
          'nodes': int(m.group(2)),
          'model': model,
          'geomean_time': geometric_mean(times) if len(times) > 1 else times[0],
          'std_time': stdev(times) if len(times) > 1 else times[0],
          'max_time': max(times),
          'min_time': min(times),
        })

    df = pd.DataFrame(data)
    path = OUT_DIR / f'dnnproxies_{sbm.get_cluster_name()}_data.csv'
    df.to_csv(path)
    print(f'Data saved to {path.resolve().absolute()}')

  print(df)

  color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
  models = df['model'].unique()
  model_color_map = dict(zip(models, itertools.cycle(color_cycle)))

  for cluster, partition in df.groupby(["cluster", "partition"]).groups.keys():
    plot_scaling_by_model(df, clusters_partitions=(cluster, partition), model_color_map=model_color_map)
    path = OUT_DIR / f'DNNProxy_{cluster}_{partition}.png'
    plt.savefig(path)
    plt.close()
    print(f'Plot saved to {path.resolve().absolute()}')

  combos = df[["cluster", "partition"]].drop_duplicates()
  pairs = [(a, b) for a in combos.itertuples(index=False, name=None) for b in combos.itertuples(index=False, name=None) if a != b]
  for (ref_cluster, ref_partition), (cmp_cluster, cmp_partition) in pairs:
    try:
      plot_performance_ratio(
        df,
        ref_cluster, ref_partition,
        cmp_cluster, cmp_partition
      )
      filename = f"DNNProxy_ratio_{ref_cluster}_{ref_partition}_vs_{cmp_cluster}_{cmp_partition}.png"
      plt.savefig(OUT_DIR / filename)
      plt.close()
      print(f"Plot saved to {filename}")

      plot_performance_ratio(
        df,
        ref_cluster, ref_partition,
        cmp_cluster, cmp_partition,
        is_barplot=True,
      )
      filename = f"DNNProxy_ratio_barplot_{ref_cluster}_{ref_partition}_vs_{cmp_cluster}_{cmp_partition}.png"
      plt.savefig(OUT_DIR / filename)
      plt.close()
      print(f"Plot saved to {filename}")
    except ValueError as e:
      print(f"Skipping {ref_cluster}/{ref_partition} vs {cmp_cluster}/{cmp_partition}: {e}")


  pairs = list(itertools.combinations(combos.itertuples(index=False, name=None), 2))
  for (ref_cluster, ref_partition), (cmp_cluster, cmp_partition) in pairs:
    try:
      plot_scaling_by_model(df, clusters_partitions=[(ref_cluster, ref_partition), (cmp_cluster, cmp_partition)], model_color_map=model_color_map)
      path = OUT_DIR / f'DNNProxy_{ref_cluster}_{ref_partition}__{cmp_cluster}_{cmp_partition}.png'
      plt.savefig(path)
      plt.close()
      print(f'Plot saved to {path.resolve().absolute()}')
    except ValueError as e:
      print(f"Skipping {ref_cluster}/{ref_partition} vs {cmp_cluster}/{cmp_partition}: {e}")



if __name__ == "__main__":
  main()
