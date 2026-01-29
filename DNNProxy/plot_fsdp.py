import matplotlib.pyplot as plt
import sys
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.utils.plots import create_color_map, create_marker_map, format_bytes

from parser import *
import argparse

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
    runs_per_rank=10,
    num_units=18,
    networks=["ib", "eth"],
    colors=None,
    networks_labels=None,
):
    """
    Plot FSDP runtime vs world_size using the same aggregation logic
    as plot_runtime_scaling (rank -> run -> world_size).
    """
    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    if networks_labels is None:
        networks_labels = {net: net for net in networks}

    filtered_df = df[df["model_name"] == model_name]

    filtered_df = filtered_df[filtered_df["num_units"] == num_units]

    if filtered_df.empty:
        print(f"No data found for model {model_name}")
        return

    plt.figure(figsize=(10, 6))

    for net in networks:
        job_df = filtered_df[filtered_df["network"] == net].copy()
        if job_df.empty:
            continue

        # ricostruzione indice di run per rank
        job_df["run_index"] = (
            job_df.groupby("rank").cumcount() % runs_per_rank
        )

        # media sui rank per ogni run
        agg_df = (
            job_df
            .groupby(["world_size", "run_index"])["runtime"]
            .mean()
            .reset_index()
        )

        # media finale sui run
        mean_runtime = (
            agg_df
            .groupby("world_size")["runtime"]
            .mean()
        )

        # ordinamento esplicito
        ws = np.array(sorted(mean_runtime.index))
        rt = mean_runtime.loc[ws].values

        label_net = networks_labels.get(net, net)
        plt.plot(
            ws,
            rt,
            "o-",
            color=colors.get(net, "gray"),
            label=label_net,
        )

    plt.xlabel("World Size")
    plt.ylabel("Time (s)")
    title_lb = f", Num Units: {num_units}" if num_units else ""
    plt.title(f"FSDP Scaling - Model: {model_name}{title_lb}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    nu_str = f"_nu{num_units}" if num_units else ""
    output_path = output_dir / f"fsdp_scaling_{model_name}{nu_str}.png"
    plt.savefig(output_path)


def _plot_comm(df, model_name, world_size, metric, msg_col, networks,
               colors, networks_labels, runs_per_rank):
    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    
    filtered_df = df[
        (df["model_name"] == model_name) &
        (df["world_size"] == world_size)
    ].copy()

    num_units_list = sorted(filtered_df["num_units"].unique())
    
    plt.figure(figsize=(10,6))
    
    for i, net in enumerate(networks):
        job_df = filtered_df[filtered_df["network"] == net].copy()
        
        # run index per rank
        job_df["run_index"] = (
            job_df.groupby("rank").cumcount() // job_df["num_units"]
        ) % runs_per_rank
        
        # 1) media su unit_idx per run
        per_run_rank = (
            job_df
            .groupby(["num_units","rank","run_index"])[metric]
            .mean()
            .reset_index()
        )
        
        # 2) media su rank per run
        per_run = (
            per_run_rank
            .groupby(["num_units","run_index"])[metric]
            .mean()
            .reset_index()
        )
        
        label_net = networks_labels[net] if networks_labels else net
        
        for j, nu in enumerate(num_units_list):
            vals = per_run[per_run["num_units"] == nu][metric].values
            x = (
                np.full_like(vals, j, dtype=float)
                + (i - 0.5) * 0.2
                + (np.random.rand(len(vals)) - 0.5) * 0.05
            )
            plt.scatter(
                x, vals,
                color=colors.get(net,"gray"),
                alpha=0.7,
                label=label_net if j == 0 else ""
            )
    
    # ✅ X-axis labels: msg size coerente con world_size
    x_labels = []
    for nu in num_units_list:
        mean_bytes = filtered_df[
            filtered_df["num_units"] == nu
        ][msg_col].mean()
        x_labels.append(
            f"{nu} units\n{format_bytes(mean_bytes, binary=True)}"
        )
    
    plt.xticks(np.arange(len(num_units_list)), x_labels)
    plt.xlabel("Units (per-rank msg size)")
    plt.ylabel(metric.replace("_"," ").title())
    plt.title(
        f"{metric.replace('_',' ').title()} per Run\n"
        f"Model: {model_name}, WS: {world_size}"
    )
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f"{metric}_scatter_{model_name}_ws{world_size}.png"
    )
    plt.close()

# Funzioni pubbliche separate
def plot_reduce_scatter(df, model_name, world_size, networks=("ib","eth"),
                        colors=None, networks_labels=None, runs_per_rank=10):
    _plot_comm(df, model_name, world_size, "reduce_scatter", "reducescatter_msg_size_bytes",
               networks, colors, networks_labels, runs_per_rank)

def plot_allgather_fwd(df, model_name, world_size, networks=("ib","eth"),
                        colors=None, networks_labels=None, runs_per_rank=10):
    _plot_comm(df, model_name, world_size, "allgather_wait_fwd", "allgather_msg_size_bytes",
               networks, colors, networks_labels, runs_per_rank)

def plot_allgather_bwd(df, model_name, world_size, networks=("ib","eth"),
                        colors=None, networks_labels=None, runs_per_rank=10):
    _plot_comm(df, model_name, world_size, "allgather_wait_bwd", "allgather_msg_size_bytes",
               networks, colors, networks_labels, runs_per_rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FSDP metrics from CSV files")
    parser.add_argument("--runtime-csv", required=True, help="Path to runtime CSV file")
    parser.add_argument("--comm-csv", required=True, help="Path to communication CSV file")
    args = parser.parse_args()

    runtime_df = pd.read_csv(args.runtime_csv)
    comm_df = pd.read_csv(args.comm_csv)
    runtime_df, comm_df = get_metrics_dataframe(strategy="fsdp")
    colors = create_color_map(["nanjing-inter", "nanjing-intra"])

    networks_labels = {
        "nanjing-inter": "nanjing-inter",
        "nanjing-intra": "nanjing-intra",
        "boost_usr_prod": "boost_usr_prod",
    }

    plot_fsdp_runtime_scaling(runtime_df, 'gpt2_l_32', 10, 18, ['nanjing-inter', 'nanjing-intra'], colors, networks_labels)

    plot_reduce_scatter(df=comm_df, model_name="gpt2_l_32", world_size=4, networks=["nanjing-inter","nanjing-intra"], networks_labels=networks_labels, colors=colors)
    plot_allgather_fwd(df=comm_df, model_name="gpt2_l_32", world_size=4, networks=["nanjing-inter","nanjing-intra"], networks_labels=networks_labels, colors=colors)
    plot_allgather_bwd(df=comm_df, model_name="gpt2_l_32", world_size=4, networks=["nanjing-inter","nanjing-intra"], networks_labels=networks_labels, colors=colors)

    plot_reduce_scatter(df=comm_df, model_name="gpt2_l_32", world_size=2, networks=["nanjing-inter","nanjing-intra"], networks_labels=networks_labels, colors=colors)
    plot_allgather_fwd(df=comm_df, model_name="gpt2_l_32", world_size=2, networks=["nanjing-inter","nanjing-intra"], networks_labels=networks_labels, colors=colors)
    plot_allgather_bwd(df=comm_df, model_name="gpt2_l_32", world_size=2, networks=["nanjing-inter","nanjing-intra"], networks_labels=networks_labels, colors=colors)

    # plot_fsdp_runtime_scaling(runtime_df, 'gpt2_l_128', 128, ['ib', 'eth'], colors, networks_labels)

    #TODO: add plots for FSDP (scaling, all-gather, reduce-scatter)