import os

import comet_ml
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import Tensor

from src.datasets import get_dataloaders
from src.models import SyntheticVAE
from src.train import train_vae

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")


def synth_loss_function(
    recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float | None = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    # MSE Loss
    MSE = F.mse_loss(recon_x, x, reduction="mean")

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return MSE + beta * KLD, MSE, KLD


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the synthetic VAE.")
    parser.add_argument("--latent-dim", type=int, default=2, help="Dimension of latent space")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Dimension of hidden layers")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    # Get data loaders
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, dataset_name="synthetic"
    )

    # Initialize model
    device = torch.device("cuda1")
    model = SyntheticVAE(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)

    # Initialize Comet experiment
    experiment = None
    experiment_name = (
        f"SynthVAE_hidden_dim_{args.hidden_dim}_"
        f"latent_dim_{args.latent_dim}_batch_size_{args.batch_size}"
    )
    if COMET_API_KEY:
        experiment = comet_ml.start(
            api_key=COMET_API_KEY,
            project_name="pdl-hw3",
            mode="create",
            online=True,
            experiment_config=comet_ml.ExperimentConfig(
                auto_metric_logging=False,
                disabled=False,
                name=experiment_name,
            ),
        )
    print(f"Starting experiment: {experiment_name}")

    # Train the model
    train_vae(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        loss_fn=synth_loss_function,
        epochs=args.epochs,
        lr=args.lr,
        experiment=experiment,
        checkpoint_name=experiment_name,
        log_every=10,  # comet log every 10 batches
    )
