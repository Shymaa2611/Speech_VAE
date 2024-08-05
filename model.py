import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, input_channels, input_length, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * (input_length // 8), hidden_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

class LatentZ(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, p_x):
        mu = self.mu(p_x)
        logvar = self.logvar(p_x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return std * eps + mu, logvar, mu
class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_channels, output_length):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64 * (output_length // 8))
        self.unflatten = nn.Unflatten(1, (64, output_length // 8))
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose1d(16, output_channels, kernel_size=4, stride=2, padding=1, output_padding=1)  # Adjusted to get the exact length

    def forward(self, z_x):
        q_x = F.relu(self.fc1(z_x))
        q_x = F.relu(self.fc2(q_x))
        q_x = self.unflatten(q_x)
        q_x = F.relu(self.deconv1(q_x))
        q_x = F.relu(self.deconv2(q_x))
        q_x = torch.sigmoid(self.deconv3(q_x))
        return q_x


class VAES(nn.Module):
    def __init__(self, input_channels, input_length, hidden_size, latent_size=2):
        super().__init__()
        self.encoder = Encoder(input_channels, input_length, hidden_size)
        self.latent_z = LatentZ(hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_channels, input_length)

    def forward(self, x):
        p_x = self.encoder(x)
        z, logvar, mu = self.latent_z(p_x)
        q_z = self.decoder(z)
        return q_z, logvar, mu, z

