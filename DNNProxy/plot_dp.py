import matplotlib.pyplot as plt
import sys
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.utils.plots import create_color_map, create_marker_map, format_bytes
from py_utils.constants.machines import *
from parser import *

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


def plot_runtime_scaling(df, model_name, bucket_size=None, local_batch_size=None, runs_per_rank=50, networks=["ib", "eth"], colors=None, networks_labels=None, linestyles=None):
    """
    Plot samples/s vs world_size for data parallel scaling. Ideal scaling is a linear increase.
    """
    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    if networks_labels is None:
        networks_labels = {net: net for net in networks}
    if linestyles is None:
        linestyles = {net: "-" for net in networks}
    
    if local_batch_size is None:
        raise ValueError("local_batch_size must be provided to calculate samples/s")
    
    filtered_df = df[df["model_name"] == model_name]
    if bucket_size is not None:
        filtered_df = filtered_df[filtered_df["num_buckets"] == bucket_size]

    plt.figure(figsize=(10,6))

    for net in networks:
        job_df = filtered_df[filtered_df["network"] == net].copy()
        if job_df.empty:
            continue  # skip networks with no data

        # runtime medio per step aggregando su rank e run
        job_df["run_index"] = job_df.groupby("rank").cumcount() % runs_per_rank
        agg_df = job_df.groupby(["world_size", "run_index"])["runtime"].mean().reset_index()
        mean_runtime = agg_df.groupby("world_size")["runtime"].mean()

        # ordinamento per sicurezza
        ws = np.array(sorted(mean_runtime.index))
        rt = mean_runtime.loc[ws].values
        
        # Calculate samples/s: (local_batch_size * world_size) / runtime
        samples_per_sec = (local_batch_size * ws) / rt

        label_net = networks_labels.get(net, net)
        plt.plot(ws, samples_per_sec, "o", linestyle=linestyles.get(net, "-"), 
                color=colors.get(net, "gray"), label=label_net, markersize=6)

    plt.xlabel("Number of nodes")
    plt.xticks(df.world_size.unique())
    plt.ylabel("Samples/s")
    plt.title(f"Data Parallelism Scaling")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    #save png to file
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_str = f"_b{bucket_size}" if bucket_size is not None else ""
    output_path = output_dir / f"dp_scaling_{model_name}{bucket_str}.png"
    plt.savefig(output_path)


def plot_barrier_scatter_by_bucket(df, model_name, world_size, networks=["ib", "eth"], colors=None, networks_labels=None, runs_per_rank=50, markers=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    if networks_labels is None:
        networks_labels = {net: net for net in networks}
    if markers is None:
        markers = {net: "o" for net in networks}

    filtered_df = df[df["model_name"] == model_name]
    buckets = sorted(filtered_df["num_buckets"].unique())

    # x-axis labels
    bucket_labels = []
    for b in buckets:
        mean_bytes = filtered_df[filtered_df["num_buckets"] == b]["msg_size_avg_bytes"].mean()
        mib = format_bytes(mean_bytes, binary=True)
        bucket_labels.append(f"{b} buckets\n{mib}")

    plt.figure(figsize=(10,6))

    for i, net in enumerate(networks):
        job_df = filtered_df[
            (filtered_df["network"] == net) &
            (filtered_df["world_size"] == world_size)
        ].copy()

        job_df["run_index"] = job_df.groupby("rank").cumcount() % runs_per_rank

        agg_df = (
            job_df.groupby(["num_buckets", "run_index"])["barrier_time"]
                  .mean()
                  .reset_index()
        )

        for j, b in enumerate(buckets):
            runs = agg_df[agg_df["num_buckets"] == b]["barrier_time"].values
            if len(runs) == 0:
                continue
            # base x = bucket index + network offset + small jitter
            x_positions = np.full_like(runs, j, dtype=float) + (i - (len(networks)-1)/2) * 0.2 + (np.random.rand(len(runs)) - 0.5) * 0.05
            plt.scatter(
                x_positions, runs,
                color=colors.get(net, "gray"),
                marker=markers.get(net, "o"),
                alpha=0.7,
                label=networks_labels[net] if j==0 else ""
            )

    # Get y-limits AFTER scatter points
    ymin, ymax = plt.ylim()

    # Draw vertical lines exactly between buckets
    for j in range(1, len(buckets)):
        plt.vlines(
            x=j - 0.5,  # halfway between buckets
            ymin=ymin,
            ymax=ymax,
            color="gray",
            linestyle="--",
            linewidth=2.0,
            alpha=0.5,
            zorder=0  # behind scatter
        )

    plt.xticks(np.arange(len(buckets)), bucket_labels)
    plt.xlabel("Buckets (Msg size x bucket)")
    plt.ylabel("Barrier Time (s)")
    plt.title(f"Barrier Time Distribution")

    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"barrier_scatter_{model_name}_ws{world_size}.png"
    plt.savefig(output_path)

    
def plot_runtime_with_barrier_stacked(
    df,
    model_name,
    bucket_size=None,
    local_batch_size=None,
    runs_per_rank=50,
    networks=["ib", "eth"],
    colors=None,
    networks_labels=None,
    hatches=None,
):
    """
    Stacked bar plot showing runtime per step vs world_size,
    where total bar height is runtime and a portion represents barrier time.
    Legend shows runtime per network and a single symbol for barrier time.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import matplotlib.patches as mpatches

    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    if networks_labels is None:
        networks_labels = {net: net for net in networks}
    if hatches is None:
        hatches = {net: "//" for net in networks}

    filtered_df = df[df["model_name"] == model_name]
    if bucket_size is not None:
        filtered_df = filtered_df[filtered_df["num_buckets"] == bucket_size]

    plt.figure(figsize=(12, 6))

    world_sizes = sorted(filtered_df["world_size"].unique())
    x = np.arange(len(world_sizes))
    width = 0.15

    for i, net in enumerate(networks):
        job_df = filtered_df[filtered_df["network"] == net].copy()
        if job_df.empty:
            continue

        # Group runs per rank
        job_df["run_index"] = job_df.groupby("rank").cumcount() % runs_per_rank

        agg_df = (
            job_df.groupby(["world_size", "run_index"])
            .agg(
                runtime=("runtime", "mean"),
                barrier_time=("barrier_time", "mean"),
            )
            .reset_index()
        )

        mean_data = agg_df.groupby("world_size")[["runtime", "barrier_time"]].mean()

        compute_times = [
            mean_data.loc[ws, "runtime"] - mean_data.loc[ws, "barrier_time"]
            if ws in mean_data.index
            else 0
            for ws in world_sizes
        ]

        barrier_times = [
            mean_data.loc[ws, "barrier_time"] if ws in mean_data.index else 0
            for ws in world_sizes
        ]

        offset = (i - len(networks) / 2) * width
        color = colors.get(net, "gray")

        # Runtime portion (bottom)
        plt.bar(
            x + offset,
            compute_times,
            width,
            color=color,
            alpha=0.8,
            label=networks_labels.get(net, net),
        )

        # Barrier portion (top)
        plt.bar(
            x + offset,
            barrier_times,
            width,
            bottom=compute_times,
            color=color,
            alpha=0.4,
            edgecolor='black',
            hatch=hatches.get(net, "//"),
            label="_nolegend_",
        )
        
        # Add percentage labels on the barrier portion
        for j, ws in enumerate(world_sizes):
            if ws in mean_data.index:
                total_runtime = mean_data.loc[ws, "runtime"]
                barrier_time = mean_data.loc[ws, "barrier_time"]
                if total_runtime > 0:
                    percentage = (barrier_time / total_runtime) * 100
                    # Position text at the middle of the barrier portion
                    y_pos = compute_times[j] + barrier_times[j] / 2
                    plt.text(
                        j + offset,
                        y_pos,
                        f'{percentage:.1f}%',
                        ha='center',
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        color='black'
                    )

    # Build clean legend
    import matplotlib.patches as mpatches
    runtime_patches = [
        mpatches.Patch(color=colors.get(net, "gray"), label=networks_labels.get(net, net))
        for net in networks
    ]
    barrier_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Barrier time')
    plt.legend(handles=runtime_patches + [barrier_patch], fontsize=10)

    plt.xlabel("Number of nodes")
    plt.xticks(x, world_sizes)
    plt.ylabel("Total runtime (s)")
    plt.title("Runtime Breakdown")
    plt.grid(True, linestyle="--", linewidth=0.5, axis="y")
    plt.tight_layout()

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_str = f"_b{bucket_size}" if bucket_size is not None else ""
    output_path = output_dir / f"runtime_barrier_stacked_{model_name}{bucket_str}.png"
    plt.savefig(output_path)


if __name__ == "__main__":
    csv_files = sys.argv[1:] if len(sys.argv) > 1 else ["metrics.csv"]
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    all_networks = ["nanjing-inter", "nanjing-intra", "ib", "eth", "boost_usr_prod"]
    
    # For scaling plot and scatter plot: use shared colors for same groups
    base_colors_grouped = create_color_map(["nanjing", "haicgu", "leo"])
    colors_grouped = {
        "nanjing-inter": base_colors_grouped["nanjing"],
        "nanjing-intra": base_colors_grouped["nanjing"],
        "ib": base_colors_grouped["haicgu"],
        "eth": base_colors_grouped["haicgu"],
        "boost_usr_prod": base_colors_grouped["leo"]
    }
    
    # For stacked plot: use unique colors for each network
    colors_unique = create_color_map(all_networks)
    
    # Define linestyles: differentiate within same color group (for scaling plot)
    linestyles = {
        "nanjing-inter": "-",      # solid
        "nanjing-intra": "--",     # dashed
        "ib": "-",                 # solid
        "eth": "--",               # dashed
        "boost_usr_prod": "-"      # solid
    }
    
    # Use create_marker_map for scatter plots
    markers = create_marker_map(all_networks)
    
    # For hatches in stacked bar - differentiate networks
    hatches = {
        "nanjing-inter": "//",
        "nanjing-intra": "\\\\",
        "ib": "//",
        "eth": "\\\\",
        "boost_usr_prod": "xx"
    }
    
    networks_labels = {
        "ib": "HAICGU-ib",
        "eth": "HAICGU-eth",
        "boost_usr_prod": "LEO-HAICGU",
        "nanjing-inter": "NJ-inter",
        "nanjing-intra": "NJ-intra"
    }

    # Scaling plot: use grouped colors + linestyles
    plot_runtime_scaling(
        df,
        model_name="vit_b_16_32",
        bucket_size=128,
        local_batch_size=32,
        runs_per_rank=10,
        networks=["nanjing-inter", "nanjing-intra", "ib", "eth", "boost_usr_prod"],
        colors=colors_grouped,
        networks_labels=networks_labels,
        linestyles=linestyles
    )

    plot_runtime_scaling(
        df,
        model_name="vit_b_16_32",
        bucket_size=8,
        local_batch_size=32,
        runs_per_rank=10,
        networks=["nanjing-inter", "nanjing-intra", "ib", "eth", "boost_usr_prod"],
        colors=colors_grouped,
        networks_labels=networks_labels,
        linestyles=linestyles
    )

    # Scatter plot: use grouped colors + markers
    plot_barrier_scatter_by_bucket(
        df,
        model_name="vit_b_16_32",
        world_size=4,
        networks=["nanjing-inter", "nanjing-intra", "ib", "eth", "boost_usr_prod"],
        colors=colors_grouped,
        networks_labels=networks_labels,
        markers=markers
    )

    # Stacked plot: use unique colors for each network
    plot_runtime_with_barrier_stacked(
        df,
        model_name="vit_b_16_32",
        bucket_size=128,
        local_batch_size=32,
        runs_per_rank=10,
        networks=["nanjing-inter", "nanjing-intra", "ib", "eth", "boost_usr_prod"],
        colors=colors_unique,
        networks_labels=networks_labels,
    )

    plot_runtime_with_barrier_stacked(
        df,
        model_name="vit_b_16_32",
        bucket_size=8,
        local_batch_size=32,
        runs_per_rank=10,
        networks=["nanjing-inter", "nanjing-intra", "ib", "eth", "boost_usr_prod"],
        colors=colors_unique,
        networks_labels=networks_labels,
    )