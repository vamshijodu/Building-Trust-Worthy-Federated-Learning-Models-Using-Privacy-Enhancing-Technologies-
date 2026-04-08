import flwr as fl
import numpy as np
import json
import os
import logging
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn
from rich.columns import Columns

from robust_aggregation import trimmed_mean

# Silence Flower logs
logging.getLogger("flwr").setLevel(logging.ERROR)

console = Console()
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
# Trust Score
# ---------------------------------------------------------
def compute_global_score(loss, accuracy):
    normalized_loss = loss / (loss + 1.0)
    return round(0.7 * accuracy + 0.3 * (1 - normalized_loss), 4)


# ---------------------------------------------------------
# Sparkline Graph
# ---------------------------------------------------------
def sparkline(data):
    ticks = "▁▂▃▄▅▆▇█"
    if not data:
        return ""
    mn, mx = min(data), max(data)
    if mn == mx:
        return ticks[0] * len(data)
    return "".join(
        ticks[int((v - mn) / (mx - mn) * (len(ticks) - 1))]
        for v in data
    )


# ---------------------------------------------------------
# Strategy with Live Rendering
# ---------------------------------------------------------
class RobustFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, rounds):
        super().__init__()
        self.rounds = rounds
        self.history = []

        self.progress = Progress(
            TextColumn("[bold cyan]Federated Training"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} Rounds"),
            console=console
        )
        self.task = self.progress.add_task("train", total=rounds)

        # Initial dashboard render
        self.renderable = self.build_dashboard()

    # -----------------------------------------------------
    # Robust Aggregation
    # -----------------------------------------------------
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None

        weights = [
            fl.common.parameters_to_ndarrays(res.parameters)
            for _, res in results
        ]

        aggregated = []
        for layer in zip(*weights):
            aggregated.append(trimmed_mean(layer))

        parameters = fl.common.ndarrays_to_parameters(aggregated)

        return parameters, {}

    # -----------------------------------------------------
    # Evaluation + Update Dashboard
    # -----------------------------------------------------
    def aggregate_evaluate(self, rnd, results, failures):

        if not results:
            return None, {}

        loss = np.mean([r.loss for _, r in results])
        accuracy = np.mean([r.metrics["accuracy"] for _, r in results])
        trust = compute_global_score(loss, accuracy)

        self.history.append({
            "round": rnd,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "trust_score": trust
        })

        with open("results/metrics.json", "w") as f:
            json.dump(self.history, f, indent=4)

        self.progress.update(self.task, advance=1)

        self.renderable = self.build_dashboard()

        return loss, {"accuracy": accuracy}

    # -----------------------------------------------------
    # Clean Dashboard Builder
    # -----------------------------------------------------
    def build_dashboard(self):

        # Table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Round", justify="center")
        table.add_column("Loss", justify="center")
        table.add_column("Accuracy (%)", justify="center")
        table.add_column("Trust", justify="center")

        for r in self.history:
            table.add_row(
                str(r["round"]),
                f"{r['loss']:.4f}",
                f"{r['accuracy']*100:.2f}",
                str(r["trust_score"])
            )

        losses = [r["loss"] for r in self.history]
        accuracies = [r["accuracy"]*100 for r in self.history]
        trust_scores = [r["trust_score"] for r in self.history]

        graph = Panel(
            f"[cyan]Loss       [/cyan]{sparkline(losses)}\n"
            f"[green]Accuracy   [/green]{sparkline(accuracies)}\n"
            f"[magenta]Trust      [/magenta]{sparkline(trust_scores)}",
            title="Metric Trends",
        )

        header = Panel(
            f"[bold cyan]Trustworthy Federated Learning[/bold cyan]\n"
            f"Robust Aggregation + Differential Privacy\n"
            f"Updated: {datetime.now().strftime('%H:%M:%S')}"
        )

        body = Columns([table, graph])

        return Panel(
            Columns([body]),
            title="Federated Learning Dashboard",
            subtitle="Live Training Monitor",
        )


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
def generate_report(history):
    with open("results/performance_report.txt", "w") as f:
        f.write("Federated Learning Performance Report\n")
        f.write("="*60 + "\n\n")
        for r in history:
            f.write(f"Round {r['round']}\n")
            f.write(f"  Loss       : {r['loss']:.4f}\n")
            f.write(f"  Accuracy   : {r['accuracy']*100:.2f}%\n")
            f.write(f"  Trust Score: {r['trust_score']}\n\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    rounds = 2
    strategy = RobustFedAvg(rounds)

    with Live(strategy.renderable, refresh_per_second=4, console=console) as live:

        # Inject live updater
        strategy.live = live

        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )

        live.update(strategy.renderable)

    generate_report(strategy.history)


if __name__ == "__main__":
    main()
