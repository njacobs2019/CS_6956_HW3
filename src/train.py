import os
import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

from datasets import get_dataloaders
from models import ConditionalVAE, ConvolutionalVAE
from utils import save_model, save_reconstructions


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


def train_epoch(model, train_loader, optimizer, device, use_conv=False):
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

    return (
        train_loss / len(train_loader.dataset),
        train_bce / len(train_loader.dataset),
        train_kld / len(train_loader.dataset),
    )


def test_epoch(model, test_loader, device, use_conv=False):
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

    return (
        test_loss / len(test_loader.dataset),
        test_bce / len(test_loader.dataset),
        test_kld / len(test_loader.dataset),
    )


def train_vae(
    latent_dim=2,
    hidden_dim=512,
    batch_size=128,
    epochs=20,
    lr=1e-3,
    use_conv=False,
    save_dir="./results",
):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    for epoch in range(1, epochs + 1):
        train_loss, train_bce, train_kld = train_epoch(
            model, train_loader, optimizer, device, use_conv
        )  # noqa: E501
        test_loss, test_bce, test_kld = test_epoch(model, test_loader, device, use_conv)

        print(f"Epoch: {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, KLD: {train_kld:.4f})")
        print(f"Test Loss: {test_loss:.4f} (BCE: {test_bce:.4f}, KLD: {test_kld:.4f})")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(
                model, optimizer, epoch, test_loss, f"{save_dir}/model_best_latent{latent_dim}.pt"
            )

            # Save reconstructions
            save_reconstructions(
                model,
                test_loader,
                device,
                f"{save_dir}/recon_latent{latent_dim}_epoch{epoch}.png",
                use_conv,
            )

        # Update learning rate
        scheduler.step()

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Save final model
    save_model(model, optimizer, epochs, test_loss, f"{save_dir}/model_final_latent{latent_dim}.pt")

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
    parser.add_argument("--save-dir", type=str, default="./results", help="Dir to save results")

    args = parser.parse_args()

    train_vae(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        use_conv=args.use_conv,
        save_dir=args.save_dir,
    )
