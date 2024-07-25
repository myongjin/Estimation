import torch
import torch.nn as nn

class Conv3DVAE(nn.Module):
    def __init__(self, input_channels, hidden_dims, latent_dim, max_length):
        super(Conv3DVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()

        # Calculate the flattened size dynamically
        test_input = torch.zeros(1, input_channels, max_length, 1, 3)
        with torch.no_grad():
            test_output = self.encoder(test_input)
        flattened_size = test_output.numel()

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(flattened_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flattened_size)
        self.hidden_dims = hidden_dims
        self.flattened_size = flattened_size  # Store for use in decode
        self.output_shape = test_output.shape[1:]  # Store shape for use in decode

        # 디코더 레이어 조정
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[2], hidden_dims[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dims[1], hidden_dims[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dims[0], input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        # print("Shape after encoder:", h.shape)
        h = self.flatten(h)
        # print("Shape after flattening:", h.shape)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z):
        h = self.fc_decode(z)
        # print("Shape before reshaping:", h.shape)
        h = h.view(-1, *self.output_shape)  # Update to match the encoder output shape
        # print("Shape after reshaping:", h.shape)
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon_x = self.decode(z)
        # print(f"recon_batch shape: {recon_x.shape}, batch shape: {x.shape}")
        return recon_x, mu, log_var
