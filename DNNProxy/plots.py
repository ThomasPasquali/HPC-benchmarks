import itertools
from pathlib import Path
import pprint
from statistics import geometric_mean
import sbatchman as sbm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union
import re

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_scaling_by_model(
  df: pd.DataFrame,
  cluster: str,
  partition: str,
  *,
  model_color_map: Union[None, Dict[str, str]] = None,
  node_col: str = "nodes",
  time_col: str = "geomean_time",
  model_col: str = "model",
  figsize: tuple = (6, 4),
  marker: str = "o"
) -> plt.Axes:
  """
  Plot scaling curves (time vs. nodes) for each model on a single figure.

  Parameters
  ----------
  df : pd.DataFrame
      The full performance DataFrame with at least columns for
      'cluster', 'partition', nodes, model, and time.
  cluster : str
      Cluster to select (e.g., "haicgu").
  partition : str
      Partition to select (e.g., "eth").
  node_col, time_col, model_col : str, optional
      Column names for the x-axis, y-axis, and model grouping.
  figsize : tuple, optional
      Size of the matplotlib figure.
  marker : str, optional
      Marker style for each line.

  Returns
  -------
  matplotlib.axes.Axes
      The axis with the plot (useful for further customization).
  """
  # ---- filter for the requested slice ----
  mask = (df["cluster"] == cluster) & (df["partition"] == partition)
  slice_df = df.loc[mask].copy()

  if slice_df.empty:
      raise ValueError(f"No data for cluster='{cluster}' & partition='{partition}'")

  # Ensure plot order by node count
  slice_df.sort_values(node_col, inplace=True)

  # ---- make the plot ----
  fig, ax = plt.subplots(figsize=figsize)

  # Plot each model separately
  for model, grp in slice_df.groupby(model_col):
    ax.plot(
      grp[node_col],
      grp[time_col],
      label=model,
      marker=marker,
      color=model_color_map[str(model)] if model_color_map else None
    )
    
  xticks = sorted(slice_df[node_col].unique())
  ax.set_xticks(xticks)
  ax.set_xlabel("Number of Nodes")
  ax.set_ylabel("Geometric Mean Time (s)")
  ax.set_title(f"Scaling on {cluster} / {partition}")
  ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
  ax.legend(title="Model")
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
  figsize: tuple = (6, 4),
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
  ax.set_ylabel("Performance Ratio (ref / cmp)")
  ax.set_title(f"Performance Ratio:\n{ref_cluster}/{ref_partition} vs. {cmp_cluster}/{cmp_partition}")
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
      })

  df = pd.DataFrame(data)
  print(df)

  color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
  models = df['model'].unique()
  model_color_map = dict(zip(models, itertools.cycle(color_cycle)))

  for cluster, partition in df.groupby(["cluster", "partition"]).groups.keys():
    plot_scaling_by_model(df, cluster=cluster, partition=partition, model_color_map=model_color_map)
    path = OUT_DIR / f'DNNProxy_{cluster}_{partition}.png'
    plt.savefig(path)
    plt.close()
    print(f'Plot saved to {path.resolve().absolute()}')

  combos = df[["cluster", "partition"]].drop_duplicates()
  pairs = list(itertools.combinations(combos.itertuples(index=False, name=None), 2))
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
        print(f"Saved: {filename}")
    except ValueError as e:
        print(f"Skipping {ref_cluster}/{ref_partition} vs {cmp_cluster}/{cmp_partition}: {e}")


if __name__ == "__main__":
  main()


# SAMPLE OUTPUTS

# [tpasquali@cn19 DNNProxy]$ mpirun -np 4 resnet 
# Rank = 2, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (allreduce) runtime for each iteration = 0.141914 s
# Rank = 3, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (allreduce) runtime for each iteration = 0.141914 s
# Rank = 0, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (allreduce) runtime for each iteration = 0.141915 s
# Rank = 1, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (allreduce) runtime for each iteration = 0.141915 s
# Rank = 2, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = 0.084589 s
# Rank = 3, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = 0.084606 s
# Rank = 0, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = 0.084609 s
# Rank = 1, world_size = 4, total_params = 25559081, ResNet-50 data parallelism (neighbors in ring) runtime for each iteration = 0.084592 s
    
# [tpasquali@cn19 DNNProxy]$ mpirun -np 6 bert 24 6
# Rank = 0, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018508 s
# Rank = 3, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018509 s
# Rank = 2, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018509 s
# Rank = 1, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018511 s
# Rank = 4, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018513 s
# Rank = 5, world_size = 6, layers = 24, stages = 6, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.018513 s

# [tpasquali@cn19 DNNProxy]$ mpirun -np 4 gpt2 
# Rank = 0, world_size = 4, layers = 48, stages = 4, total_params = 1074488320, GPT2-large pipeline and data parallelism runtime for each iteration = 0.010844 s
# Rank = 1, world_size = 4, layers = 48, stages = 4, total_params = 1074488320, GPT2-large pipeline and data parallelism runtime for each iteration = 0.010848 s
# Rank = 2, world_size = 4, layers = 48, stages = 4, total_params = 1074488320, GPT2-large pipeline and data parallelism runtime for each iteration = 0.010854 s
# Rank = 3, world_size = 4, layers = 48, stages = 4, total_params = 1074488320, GPT2-large pipeline and data parallelism runtime for each iteration = 0.010854 s

# [tpasquali@cn19 DNNProxy]$ mpirun -np 2 bert 24 2
# Rank = 0, world_size = 2, layers = 24, stages = 2, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.007392 s
# Rank = 1, world_size = 2, layers = 24, stages = 2, total_params = 367480636, Bert-large pipeline and data parallelism runtime for each iteration = 0.007392 s