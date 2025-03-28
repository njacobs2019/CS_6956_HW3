import os
import time

import comet_ml
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import optim
from torch.optim.lr_scheduler import StepLR

from datasets import get_dataloaders
from models import ConditionalVAE, ConvolutionalVAE
from utils import save_model, save_reconstructions

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


def train_epoch(model, train_loader, optimizer, device, experiment=None, step=0, use_conv=False):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0

    for batch_idx, (data, condition, _) in enumerate(train_loader):
        data, condition = data.to(device), condition.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data, condition)

        if use_conv:
            loss, bce, kld = conv_vae_loss_function(recon_batch, data, mu, logvar)
        else:
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()

        # Log metrics to Comet ML if experiment is provided
        if experiment:
            experiment.log_metric("train_loss", loss.item(), step=step)
            experiment.log_metric("train_bce", bce.item(), step=step)
            experiment.log_metric("train_kld", kld.item(), step=step)
            step += 1

    return (
        train_loss / len(train_loader.dataset),
        train_bce / len(train_loader.dataset),
        train_kld / len(train_loader.dataset),
        step,  # Return the current step for logging
    )


def test_epoch(model, test_loader, device, experiment, epoch, use_conv=False):
    model.eval()
    test_loss = 0
    test_bce = 0
    test_kld = 0

    with torch.no_grad():
        for batch_idx, (data, condition, _) in enumerate(test_loader):
            data, condition = data.to(device), condition.to(device)

            recon_batch, mu, logvar = model(data, condition)

            if use_conv:
                loss, bce, kld = conv_vae_loss_function(recon_batch, data, mu, logvar)
            else:
                loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar)

            test_loss += loss.item()
            test_bce += bce.item()
            test_kld += kld.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_bce = test_bce / len(test_loader.dataset)
    avg_test_kld = test_kld / len(test_loader.dataset)

    if experiment:
        experiment.log_metrics(
            {"test_loss": avg_test_loss, "test_bce": avg_test_bce, "test_kld": avg_test_kld},
            epoch=epoch,
        )  # Log metrics at the end of the epoch

    return avg_test_loss, avg_test_bce, avg_test_kld


def train_vae(
    latent_dim=2,
    hidden_dim=512,
    batch_size=128,
    epochs=20,
    lr=1e-3,
    use_conv=False,
    save_dir="./results",
    experiment=None,  # For Comet ML logging
):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    experiment.log_parameters(
        {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "use_conv": use_conv,
        }
    )

    # Get data loaders
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model
    if use_conv:
        model = ConvolutionalVAE(condition_dim=10, hidden_dim=hidden_dim, latent_dim=latent_dim).to(
            device
        )  # noqa: E501
    else:
        model = ConditionalVAE(
            input_dim=784, condition_dim=10, hidden_dim=hidden_dim, latent_dim=latent_dim
        ).to(device)  # noqa: E501

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    best_test_loss = float("inf")
    start_time = time.time()
    step = 0  # For Comet ML logging

    for epoch in range(1, epochs + 1):
        train_loss, train_bce, train_kld, step = train_epoch(
            model, train_loader, optimizer, device, experiment, step, use_conv
        )  # noqa: E501
        test_loss, test_bce, test_kld = test_epoch(
            model, test_loader, device, experiment, epoch, use_conv
        )

        print(f"Epoch: {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, KLD: {train_kld:.4f})")
        print(f"Test Loss: {test_loss:.4f} (BCE: {test_bce:.4f}, KLD: {test_kld:.4f})")

        experiment.log_metrics(
            {
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_kld": train_kld,
            },
            epoch=epoch,
        )

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_path = f"{save_dir}/model_best_latent{latent_dim}.pt"
            save_model(model, optimizer, epoch, test_loss, model_path)

            experiment.log_model(f"vae_best_model_latent{latent_dim}", model_path)

            recon_path = f"{save_dir}/recon_best_latent{latent_dim}_epoch{epoch}.png"
            # Save reconstructions
            save_reconstructions(
                model,
                test_loader,
                device,
                recon_path,
                use_conv,
            )

        # Update learning rate
        scheduler.step()
        experiment.log_metric("learning_rate", scheduler.get_last_lr()[0], epoch=epoch)

    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training completed in {train_time:.2f} seconds.")
    experiment.log_metric("train_time", train_time)

    # Save final model
    final_model_path = f"{save_dir}/model_final_latent{latent_dim}.pt"
    save_model(model, optimizer, epochs, test_loss, final_model_path)
    experiment.log_model(f"vae_final_model_latent{latent_dim}", final_model_path)

    experiment.end()

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Conditional VAE on MNIST.")
    parser.add_argument("--latent-dim", type=int, default=2, help="Dimension of latent space")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Dimension of hidden layers")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use-conv", action="store_true", help="Use convolutional architecture")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Dir to save results")

    args = parser.parse_args()

    experiment = None
    if COMET_API_KEY:
        experiment = comet_ml.start(
            api_key=COMET_API_KEY,
            project_name="pdl-hw3",
            mode="create",
            online=True,
            experiment_config=comet_ml.ExperimentConfig(
                auto_metric_logging=False,
                disabled=False,  # Set True for debugging runs
                name=f"{'Conv' if args.use_conv else ''}VAE-latent{args.latent_dim}-hidden{args.hidden_dim}",  # noqa: E501
            ),
        )

    train_vae(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        use_conv=args.use_conv,
        save_dir=args.save_dir,
        experiment=experiment if COMET_API_KEY else None,
    )
