import itertools
from pathlib import Path
from statistics import geometric_mean, stdev
import sys
import sbatchman as sbm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple
import re

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
  figsize: tuple = (8, 6),
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
  figsize: tuple = (8, 6),
  marker: str = "o"
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
  lines = job.get_stdout().strip().split('\n')
  times = {}

  for line in lines[1:]:
    parts = line.split(', ')
    model, time = parts[-1].split(' = ')
    time = float(time.split(' ')[0])

    if 'GPT2-' in model:
      model = 'gpt2'
    elif 'Bert-' in model:
      model = 'bert'
    elif 'ResNet-50' in model:
      if '(allreduce)' in model:
        model = 'resnet-allreduce'
      else:
        model = 'resnet-ring'

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
    jobs = filter_jobs(sbm.jobs_list(from_active=True, from_archived=True))
    data = []

    for job in jobs:
      res = parse_stdout(job)
      # print('='*50)
      # pprint.pprint(job)
      # print(res)
      for model, times in res.items():
        m = re.match(r'(\w+)_(\d+)nodes', job.config_name)
        data.append({
          'cluster': job.cluster_name,
          'partition': m.group(1),
          'nodes': int(m.group(2)),
          'model': model,
          'geomean_time': geometric_mean(times),
          'std_time': stdev(times),
          'max_time': max(times),
          'min_time': min(times),
        })

    df = pd.DataFrame(data)
    df.to_csv(OUT_DIR / 'data.csv')

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
