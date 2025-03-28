import os

import comet_ml
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from src.datasets import get_dataloaders
from src.models import ConditionalVAE, ConvolutionalVAE
from src.train import train_vae

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Binary Cross Entropy loss
    # for binary data (e.g., MNIST images)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return BCE + beta * KLD, BCE, KLD


def conv_vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Binary Cross Entropy loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return BCE + beta * KLD, BCE, KLD


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Conditional VAE on MNIST.")
    parser.add_argument("--latent-dim", type=int, default=2, help="Dimension of latent space")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Dimension of hidden layers")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use-conv", action="store_true", help="Use convolutional architecture")

    args = parser.parse_args()

    # Get data loaders
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_conv:
        model = ConvolutionalVAE(
            condition_dim=10,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        ).to(device)
    else:
        model = ConditionalVAE(
            input_dim=784,
            condition_dim=10,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        ).to(device)

    # loss function based on model type
    loss_fn = conv_vae_loss_function if args.use_conv else vae_loss_function

    # Initialize Comet experiment
    experiment = None
    if COMET_API_KEY:
        experiment = comet_ml.start(
            api_key=COMET_API_KEY,
            project_name="pdl-hw3",
            mode="create",
            online=True,
            experiment_config=comet_ml.ExperimentConfig(
                auto_metric_logging=False,
                disabled=False,
                name=f"{'Conv' if args.use_conv else ''}VAE-latent{args.latent_dim}-hidden{args.hidden_dim}",
            ),
        )

    # Train the model
    train_vae(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        epochs=args.epochs,
        lr=args.lr,
        experiment=experiment,
        checkpoint_name = f"mnist_{'conv' if args.use_conv else ''}latent{args.latent_dim}_hidden{args.hidden_dim}",  # noqa: E501
        log_every=10  # comet log every 10 batches
    )
