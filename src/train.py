import time
from pathlib import Path

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from .utils import save_model, save_reconstructions


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    loss_fn,
    experiment=None,
    step=0,
    use_conv=False,
    log_every=10,
):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0

    for batch_idx, (data, condition, _) in enumerate(train_loader):
        data, condition = data.to(device), condition.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data, condition)

        loss, bce, kld = loss_fn(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()

        # Log metrics to Comet ML
        # Only log every `log_every` batches to reduce the number of logs
        if experiment and batch_idx % log_every == 0:
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


def test_epoch(model, test_loader, device, loss_fn, experiment, epoch, use_conv=False):
    model.eval()
    test_loss = 0
    test_bce = 0
    test_kld = 0

    with torch.no_grad():
        for batch_idx, (data, condition, _) in enumerate(test_loader):
            data, condition = data.to(device), condition.to(device)

            recon_batch, mu, logvar = model(data, condition)

            loss, bce, kld = loss_fn(recon_batch, data, mu, logvar)

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
    model,
    train_loader,
    test_loader,
    device,
    loss_fn,
    epochs=20,
    lr=1e-3,
    experiment=None,
    checkpoint_name="sample_model",
    log_every=10,  # Log every `log_every` batches
):
    checkpoints_dir = "./checkpoints"
    if not Path(checkpoints_dir).exists():
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    figures_dir = "./figures"
    if not Path(figures_dir, checkpoint_name).exists():
        figures_dir = Path(figures_dir) / checkpoint_name
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

    print(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    if experiment:
        experiment.log_parameter("total_parameters", total_params)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    best_test_loss = float("inf")
    start_time = time.time()
    step = 0  # For Comet ML logging

    for epoch in range(1, epochs + 1):
        # Training
        train_loss, train_bce, train_kld, step = train_epoch(
            model, train_loader, optimizer, device, loss_fn, experiment, step, log_every
        )

        # Validation
        test_loss, test_bce, test_kld = test_epoch(
            model, test_loader, device, loss_fn, experiment, epoch
        )

        print(f"Epoch: {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, KLD: {train_kld:.4f})")
        print(f"Test Loss: {test_loss:.4f} (BCE: {test_bce:.4f}, KLD: {test_kld:.4f})")

        if experiment:
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
            model_path = f"{checkpoints_dir}/{checkpoint_name}.pt"
            save_model(model, optimizer, epoch, test_loss, model_path)

            """
            if experiment:
                experiment.log_model("vae_best_model", model_path)
            """

            recon_path = f"{figures_dir}/recon_best_epoch{epoch}.png"
            # Save reconstructions
            save_reconstructions(
                model,
                test_loader,
                device,
                recon_path,
            )

            if experiment:
                experiment.log_image(recon_path, name=f"reconstruction_epoch{epoch}")

        # Update learning rate
        scheduler.step()
        if experiment:
            experiment.log_metric("learning_rate", scheduler.get_last_lr()[0], epoch=epoch)

    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training completed in {train_time:.2f} seconds.")

    if experiment:
        experiment.log_metric("train_time", train_time)
        experiment.end()

    return model
