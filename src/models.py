import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        condition_dim: int = 10,
        hidden_dim: int = 512,
        latent_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        # Flatten the input
        x = x.view(-1, self.input_dim)

        # Concatenate input and condition
        x = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, c: Tensor) -> Tensor:
        # Concatenate latent and condition
        z = torch.cat([z, c], dim=1)
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))  # Sigmoid to output probabilities for binary data

    def forward(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Encode
        mu, logvar = self.encode(x, c)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z, c)

        return x_recon, mu, logvar

class BigConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, condition_dim=10, hidden_dim=512, latent_dim=2):
        super(BigConditionalVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Encoder with additional layer
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Additional layer
        self.fc1_2_bn = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder with additional layer
        self.fc2 = nn.Linear(latent_dim + condition_dim, hidden_dim // 2)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2_2 = nn.Linear(hidden_dim // 2, hidden_dim)  # Additional layer
        self.fc2_2_bn = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, c):
        # Flatten the input
        x = x.view(-1, self.input_dim)

        # Concatenate input and condition
        x = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1_bn(self.fc1(x)))
        h = F.relu(self.fc1_2_bn(self.fc1_2(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # Concatenate latent and condition
        z = torch.cat([z, c], dim=1)
        h = F.relu(self.fc2_bn(self.fc2(z)))
        h = F.relu(self.fc2_2_bn(self.fc2_2(h)))
        return torch.sigmoid(self.fc3(h))  # Sigmoid for binary data

    def forward(self, x, c):
        # Encode
        mu, logvar = self.encode(x, c)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z, c)

        return x_recon, mu, logvar

class ConvolutionalVAE(nn.Module):
    def __init__(self, condition_dim: int = 10, hidden_dim: int = 256, latent_dim: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Flatten(),
        )

        # output size after convolutions
        conv_output_size = 64 * 7 * 7

        # Condition embedding
        self.condition_embedding = nn.Linear(condition_dim, hidden_dim)
        # Fully connected layers
        self.fc_encoder = nn.Linear(conv_output_size + hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_decoder = nn.Linear(latent_dim + hidden_dim, conv_output_size)

        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 28x28
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        # Process image through convolutions
        x = self.encoder_conv(x)

        # Embed condition
        c = F.relu(self.condition_embedding(c))

        # Concatenate image features and condition
        x = torch.cat([x, c], dim=1)

        # Process through fully connected layer
        h = F.relu(self.fc_encoder(x))

        # Get latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, c: Tensor) -> Tensor:
        # Embed condition
        c = F.relu(self.condition_embedding(c))

        # Concatenate latent and condition
        z = torch.cat([z, c], dim=1)

        # Process through fully connected layer
        h = F.relu(self.fc_decoder(z))

        # Process through deconvolutions
        return self.decoder_conv(h.view(-1, 64, 7, 7))  # Output size is 1 channel, 28x28 image

    def forward(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Encode
        mu, logvar = self.encode(x, c)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z, c)

        return x_recon, mu, logvar


class SyntheticVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        condition_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        # Concatenate input and condition
        x = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, c: Tensor) -> Tensor:
        # Concatenate latent and condition
        z = torch.cat([z, c], dim=1)
        h = F.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Encode
        mu, logvar = self.encode(x, c)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z, c)

        return x_recon, mu, logvar
