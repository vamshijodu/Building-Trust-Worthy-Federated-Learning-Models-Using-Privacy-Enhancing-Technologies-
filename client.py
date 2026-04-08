import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from opacus import PrivacyEngine
from rich.console import Console
from rich.panel import Panel
import logging
import warnings

from model import CNN
from data import load_data

warnings.filterwarnings("ignore")
logging.getLogger("flwr").setLevel(logging.ERROR)

console = Console()


class DPClient(fl.client.NumPyClient):

    def __init__(self, cid):
        self.cid = int(cid)
        self.device = torch.device("cpu")

        self.model = CNN().to(self.device)
        self.trainloader, self.testloader = load_data(self.cid)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.05,
            momentum=0.9
        )

        # Real Differential Privacy
        self.privacy_engine = PrivacyEngine(secure_mode=False)

        self.model, self.optimizer, self.trainloader = \
            self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )

    # -----------------------------------------------------
    # Parameter Exchange
    # -----------------------------------------------------
    def get_parameters(self, config):
        return [
            val.cpu().detach().numpy()
            for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):
        state_dict = dict(
            zip(self.model.state_dict().keys(), parameters)
        )
        self.model.load_state_dict(
            {k: torch.tensor(v) for k, v in state_dict.items()},
            strict=True
        )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def fit(self, parameters, config):

        self.set_parameters(parameters)
        self.model.train()

        correct, total = 0, 0

        progress = tqdm(self.trainloader, desc=f"Client {self.cid}", ncols=100)

        for data, target in progress:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)

            loss.backward()
            self.optimizer.step()

            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            acc = correct / total
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.2f}%")

        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)

        console.print(Panel.fit(
            f"[bold blue]CLIENT {self.cid} - DIFFERENTIAL PRIVACY REPORT[/bold blue]\n\n"
            f"Privacy Budget (ε): {epsilon:.2f}\n"
            f"Noise Multiplier  : 1.0\n"
            f"Max Grad Norm     : 1.0\n\n"
            f"[bold green]✓ DP-SGD Applied Successfully[/bold green]",
            border_style="blue"
        ))

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)
        self.model.eval()

        correct, total, loss_total = 0, 0, 0.0

        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)

                loss_total += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        avg_loss = loss_total / len(self.testloader)

        return float(avg_loss), total, {"accuracy": accuracy}


# -----------------------------------------------------
# Client Entry
# -----------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    args = parser.parse_args()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=DPClient(args.cid).to_client(),
    )
