#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.constants import *
from py_utils.utils import create_color_map, create_marker_map, format_bytes
import py_utils.import_export as import_export

FONT_AXES = 20
FONT_TICKS = 14
FONT_LEGEND = 14

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize

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
        mask_d = (df["distance"] == dist)
        for src in df[mask_d]["src"].unique():
            mask = mask_d & (df["src"] == src)

            label = f"Src-Distance: {src:<2}-{dist}"
            if label in legend_added:
                label = None
            else:
                legend_added.add(label)

            plt.scatter(
                df.loc[mask, "size"] / 1024.0,
                df.loc[mask, "time"] * 1e6,
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
        labels=[f'[{format_bytes(max(b.left, 0), precision=1, binary=True)}, {format_bytes(b.right, precision=1, binary=True)}]' for b in bin_list],
        rotation=40,
        # fontsize=10,
    )

    # Legend
    for dist in distances:
        plt.scatter([], [], c=dist_color_map[dist], label=f"{dist}")

    plt.legend(
        title="Distance",
        loc='best',
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


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_files", type=Path, nargs='+')
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--time_quantiles", type=float, default=[1.0, 0.25, 0.3], nargs='+')
    parser.add_argument("--outdir", type=Path, default=Path("plots"))

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data, meta_df = import_export.read_multiple_from_parquet(args.parquet_files)

    for meta, df in data:
        if meta['nodes'] <= 1:
            continue
        print(meta)
        for q in args.time_quantiles:
            for leq_geq in (['leq', 'geq'] if q <= 0.95 else ['leq']):
                info = [f'{k}:{meta[k]}' for k in ['cluster', 'partition', 'nodes', 'buffer_size', 'scale', 'edgefactor']] + [f'quantile:{leq_geq}{int(q*100)}']
                base = '-'.join(info)
                title = ' - '.join(info)
                qval = df['time'].quantile(q)
                if leq_geq == 'leq':
                    df_filtered = df[df['time'] <= qval]
                else:
                    df_filtered = df[df['time'] >= qval]

                # Scatter plot
                plot_scatter(
                    df_filtered,
                    title=title,
                    outfile=args.outdir / f"{base}_scatter.png",
                )

                # Boxplots
                plot_binned_boxplots(
                    df_filtered,
                    title=title,
                    outfile=args.outdir / f"{base}_boxplot.png",
                    bins=args.bins,
                )

    print(f'Plots saved to "{Path(args.outdir).resolve().absolute()}"')

if __name__ == "__main__":
    main()
