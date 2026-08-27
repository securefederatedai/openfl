"""
Utility functions for Phi-4 model quantization and federated learning experiments.
This module contains:
- Memory tracking utilities
- Visualization functions for comparing 4-bit and 8-bit quantization
"""

# flake8: noqa: E501, E722

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.ticker import EngFormatter


def get_gpu_memory_info():
    """Get GPU memory usage information in MB."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return {"allocated": allocated, "reserved": reserved, "max_allocated": max_allocated}
        else:
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    except:
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}


class MemoryTracker:
    """Track GPU memory usage during training"""

    def __init__(self, collaborator_name, quant_type):
        self.collaborator_name = collaborator_name
        self.quant_type = quant_type
        self.timestamps = {}
        self.peak = {"allocated": 0, "reserved": 0, "max_allocated": 0}
        self.training_loss = None
        self.eval_loss = None

    def log(self, timestamp):
        """Log current memory usage at a specific timestamp"""
        self.timestamps[timestamp] = get_gpu_memory_info()

    def log_loss(self, training_loss=None, eval_loss=None):
        """Log training or evaluation loss"""
        if training_loss is not None:
            self.training_loss = training_loss
        if eval_loss is not None:
            self.eval_loss = eval_loss

    def update_peak(self):
        """Update peak memory usage values"""
        current = get_gpu_memory_info()
        self.peak["allocated"] = max(self.peak["allocated"], current["allocated"])
        self.peak["reserved"] = max(self.peak["reserved"], current["reserved"])
        self.peak["max_allocated"] = max(self.peak["max_allocated"], current["max_allocated"])

    def reset_peak(self):
        """Reset peak memory usage values"""
        self.peak = {"allocated": 0, "reserved": 0, "max_allocated": 0}

    def report(self):
        """Print memory usage report"""
        print(f"\n==== Memory Usage Report for {self.collaborator_name} ({self.quant_type}) ====")
        print("Peak Memory Usage:")
        print(f"  Allocated: {self.peak['allocated']:.2f} MB")
        print(f"  Reserved: {self.peak['reserved']:.2f} MB")
        print(f"  Max Allocated: {self.peak['max_allocated']:.2f} MB")

        print("\nMemory Usage by Stage:")
        for timestamp, mem in self.timestamps.items():
            print(f"  {timestamp}:")
            print(f"    Allocated: {mem['allocated']:.2f} MB")
            print(f"    Reserved: {mem['reserved']:.2f} MB")
            print(f"    Max Allocated: {mem['max_allocated']:.2f} MB")

        print("\nPerformance Metrics:")
        if self.training_loss is not None:
            print(f"  Training Loss: {self.training_loss:.4f}")
        if self.eval_loss is not None:
            print(f"  Evaluation Loss: {self.eval_loss:.4f}")
        print("-" * 50)

    def get_stats(self):
        """Get all statistics as a dictionary"""
        stats = {
            "peak_allocated": self.peak["allocated"],
            "peak_reserved": self.peak["reserved"],
            "peak_max_allocated": self.peak["max_allocated"],
            "quant_type": self.quant_type,
            "training_loss": self.training_loss,
            "eval_loss": self.eval_loss,
        }
        for timestamp, mem in self.timestamps.items():
            stats[f"{timestamp}_allocated"] = mem["allocated"]
            stats[f"{timestamp}_reserved"] = mem["reserved"]
            stats[f"{timestamp}_max_allocated"] = mem["max_allocated"]
        return stats


def plot_memory_metrics(flow_4bit, flow_8bit):  # NOQA: C901
    """Plot and compare memory metrics between 4-bit and 8-bit quantization."""
    try:
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("4-bit vs 8-bit Quantization Comparison", fontsize=16)

        # Colors for consistent plotting
        colors_4bit = {"Portland": "blue", "Seattle": "green"}
        colors_8bit = {"Portland": "darkblue", "Seattle": "darkgreen"}
        markers_4bit = {"Portland": "o", "Seattle": "s"}
        markers_8bit = {"Portland": "^", "Seattle": "D"}

        # Flatten the metric data for plotting
        memory_data = []
        for quant, flow in [("4-bit", flow_4bit), ("8-bit", flow_8bit)]:
            stats = flow.all_memory_stats
            for collab, rounds_data in stats.items():
                for round_name, metrics in rounds_data.items():
                    round_num = int(round_name.split("_")[1])
                    row = {
                        "Collaborator": collab,
                        "Round": round_num,
                        "Quantization": quant,
                        "Peak Memory (MB)": metrics.get("peak_max_allocated", 0),
                        "Training Loss": metrics.get("training_loss", 0),
                        "Eval Loss": metrics.get("eval_loss", 0),
                    }
                    memory_data.append(row)

        df = pd.DataFrame(memory_data)

        # Plot 1: Peak Memory Usage by Round
        axs[0, 0].set_title("Peak Memory Usage by Round")
        for quant_type in ["4-bit", "8-bit"]:
            for collab in df["Collaborator"].unique():
                subset = df[(df["Quantization"] == quant_type) & (df["Collaborator"] == collab)]
                color = colors_4bit[collab] if quant_type == "4-bit" else colors_8bit[collab]
                marker = markers_4bit[collab] if quant_type == "4-bit" else markers_8bit[collab]
                axs[0, 0].plot(
                    subset["Round"],
                    subset["Peak Memory (MB)"],
                    marker=marker,
                    linestyle="-",
                    label=f"{collab} ({quant_type})",
                    color=color,
                )

        axs[0, 0].set_xlabel("Round")
        axs[0, 0].set_ylabel("Memory (MB)")
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].yaxis.set_major_formatter(EngFormatter(unit="B"))

        # Plot 2: Training Loss by Round
        axs[0, 1].set_title("Training Loss by Round")
        for quant_type in ["4-bit", "8-bit"]:
            for collab in df["Collaborator"].unique():
                subset = df[(df["Quantization"] == quant_type) & (df["Collaborator"] == collab)]
                color = colors_4bit[collab] if quant_type == "4-bit" else colors_8bit[collab]
                marker = markers_4bit[collab] if quant_type == "4-bit" else markers_8bit[collab]
                axs[0, 1].plot(
                    subset["Round"],
                    subset["Training Loss"],
                    marker=marker,
                    linestyle="-",
                    label=f"{collab} ({quant_type})",
                    color=color,
                )

        axs[0, 1].set_xlabel("Round")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)

        # Plot 3: Eval Loss by Round
        axs[1, 0].set_title("Evaluation Loss by Round")
        for quant_type in ["4-bit", "8-bit"]:
            for collab in df["Collaborator"].unique():
                subset = df[(df["Quantization"] == quant_type) & (df["Collaborator"] == collab)]
                color = colors_4bit[collab] if quant_type == "4-bit" else colors_8bit[collab]
                marker = markers_4bit[collab] if quant_type == "4-bit" else markers_8bit[collab]
                axs[1, 0].plot(
                    subset["Round"],
                    subset["Eval Loss"],
                    marker=marker,
                    linestyle="-",
                    label=f"{collab} ({quant_type})",
                    color=color,
                )

        axs[1, 0].set_xlabel("Round")
        axs[1, 0].set_ylabel("Loss")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # Plot 4: Memory vs Loss (bubble chart)
        axs[1, 1].set_title("Memory Usage vs. Evaluation Loss")
        for quant_type in ["4-bit", "8-bit"]:
            for collab in df["Collaborator"].unique():
                subset = df[(df["Quantization"] == quant_type) & (df["Collaborator"] == collab)]
                color = colors_4bit[collab] if quant_type == "4-bit" else colors_8bit[collab]
                marker = markers_4bit[collab] if quant_type == "4-bit" else markers_8bit[collab]

                # Size proportional to round number for visual differentiation
                sizes = [100 * (r + 1) for r in subset["Round"]]

                axs[1, 1].scatter(
                    subset["Peak Memory (MB)"],
                    subset["Eval Loss"],
                    s=sizes,
                    alpha=0.7,
                    label=f"{collab} ({quant_type})",
                    color=color,
                    marker=marker,
                )

                # Add round number annotations
                for _, row in subset.iterrows():
                    axs[1, 1].annotate(
                        f"R{int(row['Round'])}",
                        (row["Peak Memory (MB)"], row["Eval Loss"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

        axs[1, 1].set_xlabel("Peak Memory (MB)")
        axs[1, 1].set_ylabel("Evaluation Loss")
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].xaxis.set_major_formatter(EngFormatter(unit="B"))

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

        # Print summary comparison
        print("\n==== Performance Summary ====\n")
        # Group by quantization and compute means
        summary = (
            df.groupby("Quantization")
            .agg({"Peak Memory (MB)": "mean", "Training Loss": "mean", "Eval Loss": "mean"})
            .reset_index()
        )

        # Calculate percentage difference
        mem_diff_pct = (
            (summary.loc[1, "Peak Memory (MB)"] - summary.loc[0, "Peak Memory (MB)"])
            / summary.loc[0, "Peak Memory (MB)"]
            * 100
        )

        eval_diff_pct = (
            (summary.loc[1, "Eval Loss"] - summary.loc[0, "Eval Loss"])
            / summary.loc[0, "Eval Loss"]
            * 100
        )

        print("Memory Usage Comparison:")
        print(f"  4-bit Avg: {summary.loc[0, 'Peak Memory (MB)']:.2f} MB")
        print(f"  8-bit Avg: {summary.loc[1, 'Peak Memory (MB)']:.2f} MB")
        print(
            f"  Difference: {abs(mem_diff_pct):.1f}% {'more' if mem_diff_pct > 0 else 'less'} memory with 8-bit"
        )

        print("\nEvaluation Loss Comparison:")
        print(f"  4-bit Avg: {summary.loc[0, 'Eval Loss']:.4f}")
        print(f"  8-bit Avg: {summary.loc[1, 'Eval Loss']:.4f}")
        print(
            f"  Difference: {abs(eval_diff_pct):.1f}% {'higher' if eval_diff_pct > 0 else 'lower'} loss with 8-bit"
        )

        loss_efficiency = (summary.loc[0, "Eval Loss"] - summary.loc[1, "Eval Loss"]) / (
            summary.loc[0, "Peak Memory (MB)"] - summary.loc[1, "Peak Memory (MB)"]
        )

        if loss_efficiency > 0:
            efficiency_msg = "8-bit provides more efficiency memory usage relative to loss"
        else:
            efficiency_msg = "4-bit provides more efficiency memory usage relative to loss"

        print(f"\nEfficiency Analysis: {efficiency_msg}")
    except ImportError:
        print(
            "Plotting requires matplotlib and pandas. Install with: pip install matplotlib pandas"
        )
    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")


def plot_loss_metrics(flow_4bit, flow_8bit):  # NOQA: C901
    """Plot training and evaluation loss metrics comparing 4-bit and 8-bit quantization"""
    # Extract and organize loss data
    loss_data = []

    # Helper function to safely convert tensor to float value
    def tensor_to_float(val):
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().float().numpy().item()
        return val

    # Process 4-bit data
    for collab, rounds_data in flow_4bit.all_memory_stats.items():
        for round_name, stats in rounds_data.items():
            round_num = int(round_name.split("_")[1]) if "_" in round_name else 0
            quant_type = stats.get("quant_type", "4bit")
            training_loss = tensor_to_float(stats.get("training_loss"))
            eval_loss = tensor_to_float(stats.get("eval_loss"))

            if training_loss is not None or eval_loss is not None:
                loss_data.append(
                    {
                        "Collaborator": collab,
                        "Round": round_name,
                        "Round Number": round_num,
                        "Training Loss": training_loss,
                        "Eval Loss": eval_loss,
                        "Quantization": quant_type,
                    }
                )

    # Process 8-bit data if provided
    if flow_8bit is not None:
        for collab, rounds_data in flow_8bit.all_memory_stats.items():
            for round_name, stats in rounds_data.items():
                round_num = int(round_name.split("_")[1]) if "_" in round_name else 0
                quant_type = stats.get("quant_type", "8bit")
                training_loss = tensor_to_float(stats.get("training_loss"))
                eval_loss = tensor_to_float(stats.get("eval_loss"))

                if training_loss is not None or eval_loss is not None:
                    loss_data.append(
                        {
                            "Collaborator": collab,
                            "Round": round_name,
                            "Round Number": round_num,
                            "Training Loss": training_loss,
                            "Eval Loss": eval_loss,
                            "Quantization": quant_type,
                        }
                    )

    loss_df = pd.DataFrame(loss_data)

    # Create a figure with subplots for loss metrics
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={"height_ratios": [1, 1]})

    # 1. Training loss across rounds (top plot)
    group_var = "Quantization" if flow_8bit else "Collaborator"

    sns.lineplot(
        x="Round Number",
        y="Training Loss",
        hue=group_var,
        data=loss_df,
        marker="o",
        sort=True,
        linewidth=3,
        markersize=10,
        ax=axes[0],
    )
    axes[0].set_title("Training Loss Across Rounds", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Round", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(title=group_var, bbox_to_anchor=(1.05, 1), loc="upper left")

    # 2. Evaluation loss across rounds (bottom plot)
    sns.lineplot(
        x="Round Number",
        y="Eval Loss",
        hue=group_var,
        data=loss_df,
        marker="o",
        sort=True,
        linewidth=3,
        markersize=10,
        ax=axes[1],
    )
    axes[1].set_title("Evaluation Loss Across Rounds", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Round", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].legend(title=group_var, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    if flow_8bit:
        print("\n==== Loss Comparison: 4-bit vs 8-bit ====\n")

        # Group by quantization and compute means
        summary = loss_df.groupby("Quantization").agg(
            {"Training Loss": ["mean", "std"], "Eval Loss": ["mean", "std"]}
        )

        print(
            f"Training Loss (4-bit): {summary.loc['4bit', ('Training Loss', 'mean')]:.4f} ± {summary.loc['4bit', ('Training Loss', 'std')]:.4f}"
        )
        print(
            f"Training Loss (8-bit): {summary.loc['8bit', ('Training Loss', 'mean')]:.4f} ± {summary.loc['8bit', ('Training Loss', 'std')]:.4f}"
        )
        print(
            f"\nEval Loss (4-bit): {summary.loc['4bit', ('Eval Loss', 'mean')]:.4f} ± {summary.loc['4bit', ('Eval Loss', 'std')]:.4f}"
        )
        print(
            f"Eval Loss (8-bit): {summary.loc['8bit', ('Eval Loss', 'mean')]:.4f} ± {summary.loc['8bit', ('Eval Loss', 'std')]:.4f}"
        )


def plot_aggregated_metrics(flow_4bit, flow_8bit):
    """Plot aggregated metrics comparing 4-bit and 8-bit quantization"""
    # Create a figure with subplots for aggregated metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Helper function to safely convert tensor to float value
    def tensor_to_float(val):
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().float().numpy().item()
        return val

    # Convert any tensor values to CPU before plotting
    loss_history_4bit = [tensor_to_float(x) for x in flow_4bit.average_loss_history]
    loss_history_8bit = [tensor_to_float(x) for x in flow_8bit.average_loss_history]
    agg_model_loss_4bit = [tensor_to_float(x) for x in flow_4bit.agg_model_loss_history]
    agg_model_loss_8bit = [tensor_to_float(x) for x in flow_8bit.agg_model_loss_history]
    local_model_loss_4bit = [tensor_to_float(x) for x in flow_4bit.local_model_loss_history]
    local_model_loss_8bit = [tensor_to_float(x) for x in flow_8bit.local_model_loss_history]

    # Setup data
    rounds = list(range(len(loss_history_4bit)))

    # Plot average loss history
    axes[0].plot(rounds, loss_history_4bit, "bo-", linewidth=2, markersize=8, label="4-bit")
    axes[0].plot(rounds, loss_history_8bit, "ro-", linewidth=2, markersize=8, label="8-bit")
    axes[0].set_title("Average Training Loss by Round", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Round", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Plot final metrics comparison
    metrics = ["Avg Training Loss", "Agg Model Loss", "Local Model Loss"]
    values_4bit = [loss_history_4bit[-1], agg_model_loss_4bit[-1], local_model_loss_4bit[-1]]
    values_8bit = [loss_history_8bit[-1], agg_model_loss_8bit[-1], local_model_loss_8bit[-1]]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = axes[1].bar(x - width / 2, values_4bit, width, label="4-bit", color="blue", alpha=0.7)
    bars2 = axes[1].bar(x + width / 2, values_8bit, width, label="8-bit", color="red", alpha=0.7)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axes[1].set_title("Final Metrics Comparison", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, rotation=15)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print percent differences
    print("\n==== Percentage Difference (8-bit vs 4-bit) ====\n")
    for i, metric in enumerate(metrics):
        pct_diff = ((values_8bit[i] - values_4bit[i]) / values_4bit[i]) * 100
        direction = "higher" if pct_diff > 0 else "lower"
        print(f"{metric}: 8-bit is {abs(pct_diff):.2f}% {direction} than 4-bit")
