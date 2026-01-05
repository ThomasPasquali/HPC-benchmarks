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


# Funzione interna per evitare duplicazioni
def _plot_comm(df, model_name, world_size, metric, msg_col, networks,
               colors, networks_labels, runs_per_rank):
    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    
    filtered_df = df[df["model_name"] == model_name].copy()
    num_units_list = sorted(filtered_df["num_units"].unique())
    
    plt.figure(figsize=(10,6))
    
    for i, net in enumerate(networks):
        job_df = filtered_df[
            (filtered_df["network"] == net) &
            (filtered_df["world_size"] == world_size)
        ].copy()
        
        # run index per rank (considera num_units)
        job_df["run_index"] = (job_df.groupby("rank").cumcount() // job_df["num_units"]) % runs_per_rank
        
        # 1) media su unit_idx per run
        per_run_rank = job_df.groupby(["num_units","rank","run_index"])[metric].mean().reset_index()
        
        # 2) media su rank per run
        per_run = per_run_rank.groupby(["num_units","run_index"])[metric].mean().reset_index()
        
        label_net = networks_labels[net] if networks_labels else net
        
        for j, nu in enumerate(num_units_list):
            vals = per_run[per_run["num_units"]==nu][metric].values
            x = np.full_like(vals, j, dtype=float) + (i-0.5)*0.2 + (np.random.rand(len(vals))-0.5)*0.05
            plt.scatter(x, vals, color=colors.get(net,"gray"), alpha=0.7,
                        label=label_net if j==0 else "")
    
    # X-axis labels: num_units + msg size media per colonna
    x_labels = []
    for nu in num_units_list:
        mean_bytes = filtered_df[filtered_df["num_units"]==nu][msg_col].mean()
        x_labels.append(f"{nu} units\n{format_bytes(mean_bytes, binary=True)}")
    
    plt.xticks(np.arange(len(num_units_list)), x_labels)
    plt.xlabel("Units (Msg size x bucket)")
    plt.ylabel(metric.replace("_"," ").title())
    plt.title(f"{metric.replace('_',' ').title()} per Run\nModel: {model_name}, WS: {world_size}")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{metric}_scatter_{model_name}_ws{world_size}.png")
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

def mock_df():
    networks = ["ib", "eth", "boost_usr_prod"]
    num_units_list = [1, 2, 4]
    msg_sizes = [8, 16, 32]  # 1-to-1 mapping con num_units

    rows = []

    for network in networks:
        for nu, msg_mb in zip(num_units_list, msg_sizes):
            msg_bytes = msg_mb * 1024 * 1024
            for rank in range(2):
                for run in range(4):
                    for unit_idx in range(nu):
                        rows.append({
                            "network": network,
                            "world_size": 2,
                            "sharding_factor": nu,
                            "num_replicas": 1,
                            "model_name": "gpt2_l_128",
                            "model_size_bytes": 340_217_856,
                            "local_batch_size": 8,
                            "num_units": nu,
                            "fwd_time_per_unit_us": 1200.0,
                            "bwd_time_per_unit_us": 1800.0,
                            "allgather_msg_size_bytes": msg_bytes,
                            "reducescatter_msg_size_bytes": msg_bytes,

                            "rank": rank,
                            "run": run,
                            "unit_idx": unit_idx,

                            "allgather_wait_fwd": np.random.uniform(
                                0.03 if network=="ib" else 0.05,
                                0.06 if network=="ib" else 0.09
                            ),
                            "allgather_wait_bwd": np.random.uniform(0.0, 3e-5),
                            "reduce_scatter": np.random.uniform(
                                0.25 if network=="ib" else 0.35,
                                0.45 if network=="ib" else 0.60
                            ),
                        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    runtime_df, comm_df = get_metrics_dataframe(strategy="fsdp")
    colors = create_color_map(["ib", "eth", "boost_usr_prod"])

    df = mock_df()

    networks_labels = {
        "ib": "HAICGU-ib",
        "eth": "HAICGU-eth",
        "boost_usr_prod": "boost_usr_prod",
    }

    plot_reduce_scatter(df=df, model_name="gpt2_l_128", world_size=2, networks=["ib","eth"], networks_labels=networks_labels)
    plot_allgather_fwd(df=df, model_name="gpt2_l_128", world_size=2, networks=["ib","eth"], networks_labels=networks_labels)
    plot_allgather_bwd(df=df, model_name="gpt2_l_128", world_size=2, networks=["ib","eth"], networks_labels=networks_labels)

    # plot_fsdp_runtime_scaling(runtime_df, 'gpt2_l_128', 128, ['ib', 'eth'], colors, networks_labels)

    #TODO: add plots for FSDP (scaling, all-gather, reduce-scatter)