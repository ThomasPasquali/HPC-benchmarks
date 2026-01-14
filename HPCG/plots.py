#!/usr/bin/env python3
"""
plots.py

Usage:
  python plots.py results1.csv results2.csv ... [--outdir plots]

Generates scaling plots comparing clusters and partitions.
"""

import argparse
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.constants.plots import *
from py_utils.constants.machines import *
from py_utils.utils.utils import query_meta_df_dict_pairs
from py_utils.utils.plots import (
    create_color_map,
    create_linestyle_map,
    create_marker_map,
)
import py_utils.import_export as import_export

plt.rc("axes", titlesize=FONT_AXES - 6)
plt.rc("axes", labelsize=FONT_AXES - 6)
plt.rc("xtick", labelsize=FONT_TICKS - 4)
plt.rc("ytick", labelsize=FONT_TICKS - 2)
plt.rc("legend", fontsize=FONT_LEGEND - 4)
plt.rc("figure", titlesize=FONT_TITLE)

KERNEL_LABEL_MAP = {
    'dotp': 'DDotP',
    'spmv': 'DSpMV',
    'mg': 'Precond.',
    'waxpby': 'WAXPBY',
}

def plot_gflops_scaling(df, outdir="results"):
    """Plot GFLOP/s scaling across nodes."""
    plt.figure(figsize=(10, 7))
    cluster_color_map = create_color_map(df.sort_values("cluster")["cluster"].unique())

    for cluster, grp_cluster in df.groupby("cluster"):
        partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())

        for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
            grp_sorted = grp_cluster_partition.sort_values(["nodes", "partition"])
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
    cluster_color_map = create_color_map(df.sort_values("cluster")["cluster"].unique())

    for cluster, grp_cluster in df.groupby("cluster"):
        partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())

        for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
            grp_sorted = grp_cluster_partition.sort_values(["nodes", "partition"])
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

def plot_kernel_runtime_breakdown(
    experiments: dict,
    ax=None,
):
    """
    Stacked bar plot:
      x = cluster_partition
      y = % of runtime
      slices = dotp, spmv, mg, waxpby

    Each kernel has a fixed color.
    Each sub-bar is annotated with its percentage.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # kernel definitions: (label, dataframe key, column)
    kernels = [
        ("dotp", "dotp", "dotp"),
        ("spmv", "spmv_halo", "spmv"),
        ("mg", "mg", "mg"),
        ("waxpby", "waxpby", "waxpby"),
    ]

    cps = list(experiments.keys())
    x = np.arange(len(cps))

    # one color per kernel
    kernel_names = [k for k, _, _ in kernels]
    kernel_colors = create_color_map(sorted(kernel_names))

    # collect absolute runtimes
    runtime = {k: [] for k in kernel_names}

    for cp in cps:
        for k, df_key, col in kernels:
            runtime[k].append(
                experiments[cp][df_key][col].sum()
            )

    totals = np.zeros(len(cps))
    for k in kernel_names:
        totals += np.array(runtime[k])

    bottom = np.zeros(len(cps))

    for k, _, _ in kernels:
        values = 100.0 * np.array(runtime[k]) / totals

        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=KERNEL_LABEL_MAP[k],
            color=kernel_colors[k],
            edgecolor="black",
            linewidth=0.4,
        )

        # annotate each stacked segment
        for i, bar in enumerate(bars):
            pct = values[i]
            if pct < 5:  # avoid clutter
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[i] + bar.get_height() / 2,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(cps, rotation=30, ha="right")
    ax.set_ylabel("Runtime [%]")
    ax.set_ylim(0, 100)
    ax.legend(title="Kernel")

    return ax


def plot_precond_breakdown(experiments: dict, colors: dict, ax=None):
    """
    Preconditioner (MG) breakdown:
      - For each MG iteration, calculate its 10 associated halo exchanges
      - Bar shows mean MG time split into computation vs communication
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))

    cps = list(experiments.keys())
    x = np.arange(len(cps))

    for i, cp in enumerate(cps):
        df_mg = experiments[cp]["mg"]
        df_halo = experiments[cp]["halo_precond"]
        
        num_mg_iters = len(df_mg)
        halo_per_mg = 10
        
        # For each MG iteration, calculate its communication overhead
        mg_times = []
        comm_times = []
        comp_times = []
        
        for mg_idx in range(num_mg_iters):
            mg_time = df_mg.iloc[mg_idx]["mg"]
            
            # Get the 10 halo exchanges for THIS MG iteration
            start_idx = mg_idx * halo_per_mg
            end_idx = start_idx + halo_per_mg
            halo_sum = df_halo.iloc[start_idx:end_idx]["exchange_halo"].sum()
            
            mg_times.append(mg_time)
            comm_times.append(halo_sum)
            comp_times.append(mg_time - halo_sum)
        
        # Calculate means
        mean_comm = np.mean(comm_times)
        mean_comp = np.mean(comp_times)
        comm_pct = 100 * mean_comm / (mean_comp + mean_comm) if (mean_comp + mean_comm) > 0 else 0
        
        # Stacked bar: computation + communication
        ax.bar(i, mean_comp, color='steelblue', 
               edgecolor='black', linewidth=0.5, 
               label='Computation' if i == 0 else '')
        ax.bar(i, mean_comm, bottom=mean_comp, 
               color='coral', edgecolor='black', linewidth=0.5,
               label='Communication' if i == 0 else '')
        
        # Annotate with communication percentage
        if comm_pct > 3:  # Only show if visible
            ax.text(i, mean_comp + mean_comm / 2, f'{comm_pct:.1f}%',
                    ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(cps, rotation=30, ha="right")
        ax.set_ylabel("Time (s)")
        # ax.set_title("Preconditioner Breakdown")
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    return ax

def _with_alpha(color, alpha):
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)

def plot_dotp_breakdown(
    experiments: dict,
    colors: dict,
    aggregate="sum",  # "mean" or "sum"
    figsize=(6, 4),
):
    cps = list(experiments.keys())
    nsys = len(cps)

    fig, axes = plt.subplots(
        1, nsys, figsize=figsize, sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, cp in zip(axes, cps):
        if cp not in colors:
            raise KeyError(f"Missing color for system '{cp}'")

        base_color = colors[cp]
        df = experiments[cp]["dotp"].copy()

        # ---- sort by rank, then dotp
        df = df.sort_values(["rank", "dotp"])

        # ---- aggregate per rank
        if aggregate == "mean":
            agg = df.groupby("rank", as_index=False).mean(numeric_only=True)
        elif aggregate == "sum":
            agg = df.groupby("rank", as_index=False).sum(numeric_only=True)
        else:
            raise ValueError("aggregate must be 'mean' or 'sum'")

        ranks = agg["rank"].values
        dotp = agg["dotp"].values
        allr = agg["dotp_allreduce"].values

        compute = np.clip(dotp - allr, 0.0, None)
        x = np.arange(len(ranks))

        # ---- stacked bars
        ax.bar(
            x,
            compute,
            bottom=allr,
            color=_with_alpha(base_color, 0.45),
            label="Compute",
        )
        ax.bar(
            x,
            allr,
            color=_with_alpha(base_color, 0.85),
            label="Allreduce",
        )

        ax.set_title(cp, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(ranks)
        ax.set_xlabel("Rank")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # ---- per-subplot legend
        ax.legend(fontsize=10, loc="best", frameon=False)

    axes[0].set_ylabel("Time")

    return fig, axes

# def plot_dotp_breakdown(
#     experiments: dict,
#     colors: dict,
#     ax=None,
# ):
#     """
#     dotp breakdown:
#       - gray bar: mean total dotp time ± std (per run)
#       - scatter: ALL recorded dotp_allreduce times
#         * color  -> cluster_partition
#         * marker -> run
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(7, 4))

#     cps = list(experiments.keys())
#     x = np.arange(len(cps))

#     # collect all runs across all systems
#     all_runs = sorted({
#         r
#         for cp in cps
#         for r in experiments[cp]["dotp"]["run"].unique()
#     })

#     marker_map = create_marker_map(all_runs)

#     legend_handles = {}

#     for i, cp in enumerate(cps):
#         df = experiments[cp]["dotp"]

#         mean = df["dotp"].mean()
#         std = df["dotp"].std()
#         vmin = df["dotp"].min()
#         vmax = df["dotp"].max()

#         # --- mean bar
#         ax.bar(
#             i,
#             mean,
#             color="lightgray",
#             edgecolor="black",
#             width=0.6,
#             zorder=1,
#         )

#         # --- min / max whiskers
#         ax.vlines(
#             i,
#             vmin,
#             vmax,
#             color="black",
#             linewidth=1.5,
#             zorder=2,
#         )
#         ax.hlines(
#             [vmin, vmax],
#             i - 0.08,
#             i + 0.08,
#             color="black",
#             linewidth=1.5,
#             zorder=2,
#         )

#         # --- std (±1σ) as inner error bar
#         ax.errorbar(
#             i,
#             mean,
#             yerr=std,
#             fmt="none",
#             ecolor="dimgray",
#             elinewidth=3,
#             capsize=6,
#             zorder=3,
#         )

#         # --- scatter: ALL dotp_allreduce samples
#         for run, sub in df.groupby("run"):
#             yvals = sub["dotp_allreduce"].values
#             jitter = (np.random.rand(len(yvals)) - 0.5) * 0.25

#             sc = ax.scatter(
#                 np.full(len(yvals), i) + jitter,
#                 yvals,
#                 color=colors[cp],
#                 marker=marker_map[run],
#                 s=35,
#                 zorder=2,
#                 label=f"Run {run}",
#             )

#             # one legend entry per run
#             if run not in legend_handles:
#                 legend_handles[run] = sc

#     ax.set_xticks(x)
#     ax.set_xticklabels(cps, rotation=30, ha="right")
#     ax.set_ylabel("Time")
#     ax.set_title("dotp runtime breakdown")

#     if len(all_runs) > 1:
#         ax.legend(
#             legend_handles.values(),
#             legend_handles.keys(),
#             title="Run",
#             fontsize=8,
#             loc="best",
#         )

#     return ax


def plot_spmv_halo_breakdown(
    experiments: dict,
    colors: dict,
    ax=None,
):
    """
    For spmv_halo:
      x = cluster_partition subdivided by halo_msg_size_bytes
      - gray bars: mean total spmv time ± std
      - scatter: per-run exchange_halo time
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    cps = list(experiments.keys())
    base_x = np.arange(len(cps))

    # collect all halo sizes (assumed consistent)
    halo_sizes = sorted({
        h
        for cp in cps
        for h in experiments[cp]["spmv_halo"]["halo_msg_size_bytes"].unique()
    })

    offsets = np.linspace(-0.3, 0.3, len(halo_sizes))
    width = 0.25 / max(1, len(halo_sizes))

    for h_idx, halo in enumerate(halo_sizes):
        for i, cp in enumerate(cps):
            df = experiments[cp]["spmv_halo"]

            sub = (
                df[df["halo_msg_size_bytes"] == halo]
                .groupby("run")
                .agg(
                    spmv_total=("spmv", "sum"),
                    halo_time=("exchange_halo", "sum"),
                )
            )

            mean = sub["spmv_total"].mean()
            std = sub["spmv_total"].std()

            x = base_x[i] + offsets[h_idx]

            ax.bar(
                x,
                mean,
                width=width,
                yerr=std,
                color="lightgray",
                edgecolor="black",
                capsize=3,
                zorder=1,
            )

            jitter = (np.random.rand(len(sub)) - 0.5) * width
            ax.scatter(
                np.full(len(sub), x) + jitter,
                sub["halo_time"],
                color=colors[cp],
                s=18,
                zorder=2,
            )

    ax.set_xticks(base_x)
    ax.set_xticklabels(cps, rotation=30, ha="right")
    ax.set_ylabel("Time")
    ax.set_title("spmv_halo runtime vs communication")

    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison scaling plots from HPCG CSVs"
    )
    parser.add_argument(
        "csv_files", nargs="+", help="Input CSV files (from parse_hpcg.py)"
    )
    parser.add_argument("--outdir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    # Create output directory
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    meta_dfs_pairs, meta_df = import_export.read_multiple_from_parquet(args.csv_files)
    import_export.describe_pairs_content(meta_dfs_pairs, verbose=True)

    print(meta_dfs_pairs[0][1]['halo_precond'].head(20))
    print(meta_dfs_pairs[0][1]['mg'].head(20))

    print(len(meta_dfs_pairs[0][1]['halo_precond']))
    print(len(meta_dfs_pairs[0][1]['mg']))
 
    if meta_df is None:
        raise Exception("meta_df is None")

    required = {
        "cluster",
        "partition",
        "nodes",
        "threads",
        "time_tot",
        "gflops",
        "gflops_opt",
        "mem",
        "global_nx",
        "global_ny",
        "global_nz",
        "num_equations",
        "final_result_valid",
    }
    if not required.issubset(meta_df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(required)}")

    # Map names and prepare dataframe
    meta_df["cluster"] = meta_df["cluster"].map(CLUSTER_NAMES_MAP)
    meta_df["partition"] = meta_df["partition"].map(PARTITION_NAMES_MAP)
    meta_df["cluster_partition"] = (
        meta_df["cluster"].astype(str) + "-" + meta_df["partition"].astype(str)
    )
    meta_df["grid"] = (
        meta_df["global_nx"].astype(str)
        + "x"
        + meta_df["global_ny"].astype(str)
        + "x"
        + meta_df["global_nz"].astype(str)
    )
    for meta, _ in meta_dfs_pairs:
        meta["cluster"] = CLUSTER_NAMES_MAP[meta["cluster"]]
        meta["partition"] = PARTITION_NAMES_MAP[meta["partition"]]
        meta["cluster_partition"] = f'{meta["cluster"]}-{meta["partition"]}'
        meta["grid"] = f'{meta["global_nx"]}x{meta["global_ny"]}x{meta["global_nz"]}'

    print("Metedata df:")
    print(meta_df)
    print("\nGenerating plots...")

    # FIXME uncomment Generate all plots
    # plot_gflops_scaling(meta_df, args.outdir)
    # plot_runtime_scaling(meta_df, args.outdir)

    grids = sorted(meta_df["grid"].unique())
    nodes_list = sorted(meta_df["nodes"].astype(int).unique())
    cluster_partitions = sorted(meta_df["cluster_partition"].unique())

    colors = create_color_map(cluster_partitions)

    # for grid in grids:
    grid = '128x128x128'
    for nodes in [4]:

        # select matching experiments
        pairs = query_meta_df_dict_pairs(
            meta_dfs_pairs,
            [("nodes", nodes)], # ("grid", grid)
        )

        if len(pairs) != len(cluster_partitions):
            warnings.warn(
                f"[grid={grid}, nodes={nodes}] "
                f"Expected {len(cluster_partitions)} experiments, "
                f"got {len(pairs)} - skipping"
            )
            continue

        # normalize into cp -> dfs dict
        experiments = {
            meta["cluster_partition"]: dfs
            for meta, dfs in pairs
        }
        # print(pairs[0][1]['dotp'])

        prefix = f"{args.outdir}/runtime_{grid}_{nodes}nodes"

        # ============================================================
        # 1) Kernel contribution stacked bar plot
        # ============================================================
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_kernel_runtime_breakdown(
            experiments=experiments,
            ax=ax,
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(f"{prefix}_kernel_breakdown.png", dpi=150)
        plt.close(fig)

        # ============================================================
        # 2) dotp breakdown
        # ============================================================
        fig, ax = plot_dotp_breakdown(
            experiments=experiments,
            colors=colors,
        )
        fig.tight_layout()
        fig.savefig(f"{prefix}_dotp_breakdown.png", dpi=150)
        plt.close(fig)

        # ============================================================
        # 3) spmv_halo breakdown
        # ============================================================
        # fig, ax = plt.subplots(figsize=(8, 4))
        # plot_spmv_halo_breakdown(
        #     experiments=experiments,
        #     colors=colors,
        #     ax=ax,
        # )
        # fig.tight_layout()
        # fig.savefig(f"{prefix}_spmv_halo_breakdown.png", dpi=150)
        # plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4))
        plot_precond_breakdown(
            experiments=experiments,
            colors=colors,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(f"{prefix}_precond_breakdown.png", dpi=150)
        plt.close(fig)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
