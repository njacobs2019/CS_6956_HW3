import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def save_model(
    model: nn.Module, optimizer: Optimizer, epoch: float, loss: float, filepath: str
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        filepath,
    )
    print(f"Model saved to {filepath}")


def load_model(
    model: nn.Module, optimizer: Optimizer, filepath: str, device: torch.device
) -> tuple[nn.Module, Optimizer, int, float]:
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss


def save_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    filepath: str,
    use_conv: bool = False,
) -> None:
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        data, condition, label = next(iter(dataloader))
        data, condition = data.to(device), condition.to(device)

        # Get reconstructions
        recon_batch, mu, logvar = model(data, condition)

        # Reshape for visualization if using non-convolutional model
        if not use_conv:
            recon_batch = recon_batch.view(-1, 1, 28, 28)

        # Create comparison grid
        n = min(8, data.size(0))
        comparison = torch.cat([data[:n], recon_batch[:n]])

        # Save the grid
        save_image(comparison, filepath, nrow=n)
        print(f"Reconstructions saved to {filepath}")


def save_samples(
    model: nn.Module,
    device: torch.device,
    filepath: str,
    num_samples: int = 64,
    latent_dim: int = 2,
    condition_dim: int = 10,
    use_conv: bool = False,
) -> None:
    model.eval()
    with torch.no_grad():
        # Create one-hot encoded conditions for each digit
        samples_per_digit = num_samples // 10
        conditions = []

        for digit in range(10):
            condition = torch.zeros(samples_per_digit, condition_dim)
            condition[:, digit] = 1.0
            conditions.append(condition)

        condition = torch.cat(conditions, dim=0).to(device)

        # Sample from latent space
        z = torch.randn(condition.size(0), latent_dim).to(device)

        # Generate samples
        samples = model.decode(z, condition)

        if not use_conv:
            samples = samples.view(-1, 1, 28, 28)

        # Save samples
        save_image(samples, filepath, nrow=10)
        print(f"Samples saved to {filepath}")


def save_image(tensor: Tensor, filepath: str, nrow: int = 8) -> None:
    grid = make_grid(tensor, nrow=nrow, padding=2, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis("off")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def interpolate_latent_space(
    model: nn.Module,
    device: torch.device,
    filepath: str,
    start_digit: int = 1,
    end_digit: int = 7,
    steps: int = 10,
    use_conv: bool = False,
) -> None:
    model.eval()
    with torch.no_grad():
        # Create conditions for start and end digits
        condition_start = torch.zeros(1, 10)
        condition_start[0, start_digit] = 1.0
        condition_start = condition_start.to(device)

        condition_end = torch.zeros(1, 10)
        condition_end[0, end_digit] = 1.0
        condition_end = condition_end.to(device)

        # Sample from latent space for each digit
        z_start = torch.randn(1, model.latent_dim).to(device)
        z_end = torch.randn(1, model.latent_dim).to(device)

        # Generate interpolated points in latent space
        z_interpolated = torch.zeros(steps, model.latent_dim).to(device)

        for i in range(steps):
            alpha = i / (steps - 1)
            z_interpolated[i] = (1 - alpha) * z_start + alpha * z_end

        # Generate samples for interpolated latent points with fixed condition (start digit)
        samples_z = model.decode(z_interpolated, condition_start.repeat(steps, 1))

        # Generate samples for fixed latent point (start) with interpolated conditions
        condition_interpolated = torch.zeros(steps, 10).to(device)

        for i in range(steps):
            alpha = i / (steps - 1)
            condition_interpolated[i] = (1 - alpha) * condition_start + alpha * condition_end

        samples_c = model.decode(z_start.repeat(steps, 1), condition_interpolated)

        # Reshape for visualization if using non-convolutional model
        if not use_conv:
            samples_z = samples_z.view(-1, 1, 28, 28)
            samples_c = samples_c.view(-1, 1, 28, 28)

        # Save the interpolations
        save_image(samples_z, f"{filepath}_z_interp_{start_digit}to{start_digit}.png", nrow=steps)
        save_image(samples_c, f"{filepath}_c_interp_{start_digit}to{end_digit}.png", nrow=steps)
        print(f"Interpolations saved to {filepath}")


def visualize_latent_space(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    filepath: str,
    use_conv: bool = False,
) -> None:
    model.eval()
    with torch.no_grad():
        # Only visualize if latent space is 2D
        if model.latent_dim != 2:
            print(
                "Latent space visualization only supported for 2D latent space, "
                f"got {model.latent_dim}D"
            )
            return

        # Store encoded data points and their labels
        z_points = []
        labels = []

        for batch_idx, (data, condition, label) in enumerate(dataloader):
            # Limit to a reasonable number of points
            if batch_idx * data.size(0) >= 1000:
                break

            data, condition = data.to(device), condition.to(device)

            # Encode data
            mu, _ = model.encode(data, condition)

            # Store the points and labels
            z_points.append(mu.cpu().numpy())
            labels.append(label.numpy())

        # Concatenate all batches
        z_points = np.concatenate(z_points, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Plot latent space
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap="tab10", alpha=0.6)
        plt.colorbar(scatter, label="Digit")
        plt.title("Latent Space Visualization")
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(filepath)
        plt.close()
        print(f"Latent space visualization saved to {filepath}")
