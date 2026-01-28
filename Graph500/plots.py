#!/usr/bin/env python3
import sys
import warnings
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

sys.path.append(str(Path(__file__).parent.parent / "machines"))
from Leonardo.nodelists_generator import LeonardoNodelistGenerator
from HAICGU.nodes_map import HAICGUNodesMap
from Nanjing.nodes_map import NanjingNodesMap

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.constants.plots import *
from py_utils.constants.machines import *
from py_utils.utils.plots import (
    create_color_map,
    create_linestyle_map,
    create_marker_map,
    format_bytes,
    parse_bytes,
)
from py_utils.utils.utils import query_meta_df_dict_pairs
import py_utils.import_export as import_export

FONT_AXES = 20
FONT_TICKS = 14
FONT_LEGEND = 12

plt.rc("axes", titlesize=FONT_AXES - 6)
plt.rc("axes", labelsize=FONT_AXES)
plt.rc("xtick", labelsize=FONT_TICKS)
plt.rc("ytick", labelsize=FONT_TICKS)
plt.rc("legend", fontsize=FONT_LEGEND)
plt.rc("figure", titlesize=FONT_TITLE - 6)

FIG_SIZE_SCALING = (10, 5)

EXCLUDED_IMPLEMENTATIONS = []
REMOVE_OUTLIERS = False
FORCE_Y_LIMS = False


# ---------------------------
# SCATTERPLOT
# ---------------------------
def plot_scatter(df: pd.DataFrame, title: str, outfile: Path):
    plt.figure(figsize=(12, 6))

    df = df.sort_values("size")
    distances = sorted(df["distance"].unique())
    sources = sorted(df["src"].unique())

    distance_color_map = create_color_map(distances)
    src_marker_map = create_marker_map(sources)

    legend_added = set()

    for dist in sorted(distances, reverse=True):
        mask_d = df["distance"] == dist
        for src in df[mask_d]["src"].unique():
            mask = mask_d & (df["src"] == src)

            label = f"Src-Distance: {src:<2}-{dist}"
            if label in legend_added:
                label = None
            else:
                legend_added.add(label)

            plt.scatter(
                df.loc[mask, "size"].astype(float) / 1024.0,
                df.loc[mask, "time"].astype(float) * 1e6,
                c=distance_color_map[dist],
                marker=src_marker_map[src],
                alpha=0.25,
                label=label,
            )

    plt.xlabel("Packet Size [KiB]")
    plt.ylabel("Time [µs]")
    plt.title(title, fontsize=12)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=7,
        ncols=1,
        frameon=True,
    )
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


# ---------------------------
# BOX-PLOTS binned by size
# ---------------------------
def plot_binned_boxplots(df: pd.DataFrame, title: str, outfile: Path, bins=6):
    df = df.copy()

    df["size_bin"] = pd.cut(df["size"], bins=bins)

    distances = sorted(df["distance"].unique())
    dist_color_map = create_color_map(distances)

    plt.figure(figsize=(10, 6))

    positions = []
    data_per_position = []
    box_colors = []

    bin_list = list(sorted(df["size_bin"].dropna().unique()))

    for i, bin_interval in enumerate(bin_list):
        group = df[df["size_bin"] == bin_interval]

        for j, dist in enumerate(distances):
            g = group[group["distance"] == dist]["time"] * 1e6
            if g.empty:
                continue

            pos = i + j * 0.12  # offset per distance
            positions.append(pos)
            data_per_position.append(g)
            box_colors.append(dist_color_map[dist])

    bp = plt.boxplot(
        data_per_position,
        positions=positions,
        widths=0.1,
        patch_artist=True,
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xticks(
        ticks=[i for i in range(len(bin_list))],
        labels=[
            f"[{format_bytes(max(b.left, 0), precision=0, binary=True)}, {format_bytes(b.right, precision=0, binary=True)}]"
            for b in bin_list
        ],
        rotation=40,
        # fontsize=10,
    )

    # Legend
    for dist in distances:
        plt.scatter([], [], c=dist_color_map[dist], label=f"{dist}")

    plt.legend(
        title="Distance",
        loc="best",
        # loc="upper left",
        # bbox_to_anchor=(1.02, 1.0),
        # fontsize=6,
        frameon=True,
    )

    plt.ylabel("Time [$\\mu$s]")
    plt.xlabel("Message Size")
    plt.title(title, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def plot_scaling(df: pd.DataFrame, outdir: Path):
    cluster_partition_linestyle_map = create_linestyle_map(
        (df["cluster_partition"]).sort_values().unique()
    )
    buffer_size_color_map = create_color_map(df["buffer_size"].sort_values().unique())

    all_vals = []
    for _, group in df.groupby(["scale", "edgefactor"]):
        for _, buffer_size_group in group.groupby(["cluster_partition", "buffer_size"]):
            buffer_size_group_sorted = buffer_size_group.sort_values("nodes")
            all_vals.extend(buffer_size_group_sorted["teps"].values)
    all_vals = [v / 1e6 for v in all_vals]
    ymin, ymax = min(all_vals), max(all_vals)
    y_pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ymin -= y_pad
    ymax += y_pad
    max_cluster_partition_len = df["cluster_partition"].map(lambda s: len(s)).max()

    # Plot TEPS
    for (scale, ef), group in df.groupby(["scale", "edgefactor"]):
        plt.figure(figsize=FIG_SIZE_SCALING)

        x_ticks_nodes = set()
        for (buffer_size_raw, cluster, partition), group in group.groupby(
            ["buffer_size_raw", "cluster", "partition"]
        ):
            group_sorted = group.sort_values(["nodes", "buffer_size_raw"])
            nodes = group_sorted["nodes"]
            teps_vals = group_sorted["teps"]
            buffer_size = format_bytes(
                buffer_size_raw,
                precision=0,
                binary=True,
                space_between_size_and_unit=False,
            )

            color = buffer_size_color_map[buffer_size]
            linestyle = cluster_partition_linestyle_map[f"{cluster}-{partition}"]

            cp = f"{cluster}-{partition}"
            plt.plot(
                nodes,
                teps_vals / 1e6,
                color=color,
                marker="o",
                markersize=6,
                linestyle=linestyle,
                label=f"{cp:<{max_cluster_partition_len}} {buffer_size}",
            )
            x_ticks_nodes |= set(nodes.values)

        if FORCE_Y_LIMS:
            plt.ylim(ymin, ymax)
        plt.title(f"Graph500 Scaling - Scale={scale}, Edgefactor={ef}")
        plt.xlabel("Nodes")
        plt.ylabel("GTEPS")
        plt.xticks(sorted(list(x_ticks_nodes)))
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(title="Cluster-Partition BufferSize", prop={"family": "monospace"})
        plt.tight_layout()
        path = outdir / f"Graph500_scaling_s{scale}_ef{ef}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()


def plot_barrier_time(nodes, scale, edgefactor, experiments: dict, outdir: Path):
    """
    Plot mean barrier time as a boxplot over buffer sizes, grouped by cluster_partition.

    Parameters:
    - nodes, scale, edgefactor: int, graph parameters
    - experiments: dict mapping (cluster_partition, buffer_size) -> pd.DataFrame
    - outdir: Path to save plots
    """

    # Helper: remove outliers using IQR
    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

    # Concatenate all experiments and inject metadata (buffer_size)
    dfs = []
    for (cluster_partition, buffer_size), df in experiments.items():
        df_copy = df.copy()
        df_copy["cluster_partition"] = cluster_partition
        df_copy["buffer_size"] = buffer_size
        dfs.append(df_copy)
    df_full = pd.concat(dfs, ignore_index=True)

    # Optionally remove outliers
    subset = df_full
    if REMOVE_OUTLIERS:
        subset = remove_outliers_iqr(df_full, "time")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.title(f"Nodes: {nodes}, Scale: {scale}, Edgefactor: {edgefactor}", fontsize=16)

    sns.boxplot(
        data=subset,
        x="buffer_size",
        y="time",
        hue="cluster_partition",
        showfliers=not REMOVE_OUTLIERS,
    )
    plt.xlabel("Buffer Size")
    plt.ylabel("Avg Barrier Time [s]")
    plt.legend(title="Cluster/Partition")
    plt.tight_layout()

    # Save
    path = (
        outdir
        / f'Graph500_barrier{"_noutliers" if REMOVE_OUTLIERS else ""}_s{scale}_ef{edgefactor}_n{nodes}.png'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f"Plot saved to {path}")
    plt.close()


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_files", type=Path, nargs="+")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument(
        "--time_quantiles", type=float, default=[1.0, 0.25, 0.3], nargs="+"
    )
    parser.add_argument("--outdir", type=Path, default=Path("plots"))

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    meta_df_dict_pairs, meta_df = import_export.read_multiple_from_parquet(
        args.parquet_files
    )
    import_export.describe_pairs_content(meta_df_dict_pairs, verbose=False)

    if meta_df is None:
        raise Exception("meta_df is None")
    
    # Remove old data
    meta_df = meta_df[~meta_df["buffer_size"].str.contains("buf")]
    print(meta_df)

    meta_df["cluster"] = meta_df["cluster"].map(CLUSTER_NAMES_MAP)
    meta_df["partition"] = meta_df["partition"].map(PARTITION_NAMES_MAP)
    meta_df["cluster_partition"] = (
        meta_df["cluster"].astype(str) + "-" + meta_df["partition"].astype(str)
    )
    meta_df["buffer_size_raw"] = (
        meta_df["buffer_size"].map(lambda x: parse_bytes(x, True)).astype(int)
    )

    # Collect indices to keep
    indices_to_keep = []
    leo_gen = LeonardoNodelistGenerator()
    
    for idx, (meta, df_dict) in enumerate(meta_df_dict_pairs):
        if 'buf' in meta['buffer_size']:
            continue
        meta["cluster"] = CLUSTER_NAMES_MAP[meta["cluster"]]
        meta["partition"] = PARTITION_NAMES_MAP[meta["partition"]]
        meta["cluster_partition"] = f'{meta["cluster"]}-{meta["partition"]}'
        cluster_name = meta["cluster"]
        if not meta.get("rank_node_map"):
            continue
        ranks_nodes_map = meta["rank_node_map"]
        if df_dict.get("packets") is None:
            continue
        df = df_dict["packets"]
        indices_to_keep.append(idx)

        gen = None
        if cluster_name == CLUSTER_NAMES_MAP["leonardo"]:
            gen = leo_gen
        elif cluster_name == CLUSTER_NAMES_MAP["haicgu"]:
            gen = HAICGUNodesMap()
        elif cluster_name == CLUSTER_NAMES_MAP["nanjing"]:
            gen = NanjingNodesMap()
        # Add here more distances scripts
        
        if gen is not None:
            df["distance"] = df.apply(
                lambda row: gen.get_node_distance(
                    ranks_nodes_map[str(int(row["src"]))],
                    ranks_nodes_map[str(int(row["dest"]))],
                ),
                axis=1,
            )
        else:
            warnings.warn(
                f"No distance script found for cluster {cluster_name}. Skipping (distances set to -1)."
            )
            df["distance"] = -1
            continue
    
    # Filter meta_df_dict_pairs
    meta_df_dict_pairs = [meta_df_dict_pairs[i] for i in indices_to_keep]
    
    # print(meta_df)
    # exit()

    Path(args.outdir / "scatter").mkdir(parents=True, exist_ok=True)
    Path(args.outdir / "boxplot").mkdir(parents=True, exist_ok=True)
    Path(args.outdir / "barrier").mkdir(parents=True, exist_ok=True)
    plot_scaling(meta_df, args.outdir)

    for (nodes, scale, ef), _ in meta_df.groupby(["nodes", "scale", "edgefactor"]):
        if nodes <= 1:
            continue
        pairs = query_meta_df_dict_pairs(
            meta_df_dict_pairs, [("nodes", nodes), ("scale", scale), ("edgefactor", ef)]
        )
        experiments = {
            (meta["cluster_partition"], meta["buffer_size"]): dfs["barrier"]
            for meta, dfs in pairs
        }

        # Barrier plot
        plot_barrier_time(nodes, scale, ef, experiments, outdir=args.outdir / "barrier")

    ## TODO implement a filter
    for meta, df in meta_df_dict_pairs:
        if meta["nodes"] <= 1:
            continue
        df = df["packets"]
        for q in args.time_quantiles:
            for leq_geq in ["leq", "geq"] if q <= 0.95 else ["leq"]:
                info = [
                    f"{k}:{meta[k]}"
                    for k in [
                        "cluster",
                        "partition",
                        "nodes",
                        "buffer_size",
                        "scale",
                        "edgefactor",
                    ]
                ] + [f"quantile:{leq_geq}{int(q*100)}"]
                base = "-".join(info)
                title = " - ".join(info)
                qval = df["time"].quantile(q)
                if leq_geq == "leq":
                    df_filtered = df[df["time"] <= qval]
                else:
                    df_filtered = df[df["time"] >= qval]

                # Scatter plot
                plot_scatter(
                    df_filtered,
                    title=title,
                    outfile=args.outdir / "scatter" / f"{base}_scatter.png",
                )

                # Boxplots
                plot_binned_boxplots(
                    df_filtered,
                    title=title,
                    outfile=args.outdir / "boxplot" / f"{base}_boxplot.png",
                    bins=args.bins,
                )

    print(f'Plots saved to "{Path(args.outdir).resolve().absolute()}"')


if __name__ == "__main__":
    main()
