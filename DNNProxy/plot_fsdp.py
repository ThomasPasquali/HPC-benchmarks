import matplotlib.pyplot as plt
import sys
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.utils.plots import create_color_map, create_marker_map, format_bytes

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

def plot_fsdp_runtime_scaling(
    df,
    model_name,
    local_batch_size=None,
    networks=["ib", "eth"],
    colors=None,
    networks_labels=None,
):
    """
    Plot FSDP runtime vs world_size using mean runtime only (no variance).
    """
    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    if networks_labels is None:
        networks_labels = {net: net for net in networks}

    # Filter by model and optionally local batch size
    filtered_df = df[df["model_name"] == model_name]
    if local_batch_size is not None:
        filtered_df = filtered_df[filtered_df["local_batch_size"] == local_batch_size]

    if filtered_df.empty:
        print(
            f"No data found for model {model_name} "
            f"with local_batch_size={local_batch_size}"
        )
        return

    plt.figure(figsize=(10, 6))

    for net in networks:
        job_df = filtered_df[filtered_df["network"] == net]
        if job_df.empty:
            continue

        # Aggregate mean runtime across all ranks and runs
        stats = (
            job_df.groupby("world_size")["runtime"]
            .mean()
            .reset_index()
        )

        ws = stats["world_size"].values
        mean_rt = stats["runtime"].values

        label_net = networks_labels.get(net, net)
        plt.plot(
            ws,
            mean_rt,
            "o-",
            color=colors.get(net, "gray"),
            label=label_net,
        )

    plt.xlabel("World Size")
    plt.ylabel("Runtime")
    title_lb = f", Local Batch: {local_batch_size}" if local_batch_size else ""
    plt.title(f"FSDP Scaling - Model: {model_name}{title_lb}")
    plt.xticks(ws)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    lbs_str = f"_lbs{local_batch_size}" if local_batch_size else ""
    output_path = output_dir / f"fsdp_scaling_{model_name}{lbs_str}.png"
    plt.savefig(output_path)

    
if __name__ == "__main__":
    runtime_df, comm_df = get_metrics_dataframe(strategy="fsdp")
    colors = create_color_map(["ib", "eth", "boost_usr_prod"])

    networks_labels = {
        "ib": "HAICGU-ib",
        "eth": "HAICGU-eth",
        "boost_usr_prod": "boost_usr_prod",
    }

    plot_fsdp_runtime_scaling(runtime_df, 'gpt2_l_128', 128, ['ib', 'eth'], colors, networks_labels)

    #TODO: add plots for FSDP (scaling, all-gather, reduce-scatter)