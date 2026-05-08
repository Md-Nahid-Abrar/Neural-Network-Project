"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ## 🏗️ Task 2 — Variational Autoencoder (Multi-Genre)
# ### Architecture
# ──────────────────────────────────────────────────────────────

# %%
# @title Task 2 — VAE architecture"""
class VAEEncoder(nn.Module):
    """
    Encoder: (T, 88) → (μ, log σ²) ∈ R^latent_dim
    Both μ and log σ² produced by separate linear heads on top of LSTM.
    """
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm   = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv  = nn.Linear(hidden_dim, latent_dim)   # log-variance

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]                                       # last-layer hidden
        mu     = self.fc_mu(h)
        log_var = self.fc_lv(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """Decoder identical to Task 1 decoder (raw logits out)."""
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=88,
                 seq_len=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in   = nn.Linear(latent_dim, latent_dim)
        self.lstm    = nn.LSTM(latent_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        self.fc_out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z_rep  = self.fc_in(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_rep)
        return self.fc_out(out)             # raw logits (B, T, 88)


class MusicVAE(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64,
                 seq_len=WIN_LEN, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim,
                                  num_layers, dropout)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim,
                                  seq_len, num_layers, dropout)
        self.latent_dim = latent_dim

    def reparameterise(self, mu, log_var):
        """z = μ + σ · ε,  ε ~ N(0, I)"""
        std = torch.exp(0.5 * log_var)           # σ = exp(0.5 · log σ²)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterise(mu, log_var)
        logits      = self.decoder(z)
        return logits, mu, log_var

    def sample(self, n=1):
        z = torch.randn(n, self.latent_dim).to(next(self.parameters()).device)
        return torch.sigmoid(self.decoder(z))    # probabilities


def vae_loss(logits, targets, mu, log_var, beta=1.0,
             focal_alpha=0.75, focal_gamma=2.0):
    """
    L_VAE = L_recon  +  β · KL(q(z|x) || p(z))
    KL closed form: -0.5 * sum(1 + log σ² - μ² - σ²)
    """
    # Reconstruction (Focal Loss)
    bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt    = torch.where(targets == 1, probs, 1 - probs)
    at    = torch.where(targets == 1,
                        torch.tensor(focal_alpha, device=logits.device),
                        torch.tensor(1 - focal_alpha, device=logits.device))
    recon_loss = (at * (1 - pt) ** focal_gamma * bce).mean()

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

print("✅ Task 2 VAE architecture defined.")
_vae = MusicVAE().to(DEVICE)
_x   = torch.zeros(4, WIN_LEN, 88).to(DEVICE)
_lo, _mu, _lv = _vae(_x)
print(f"   Input: {_x.shape} → μ: {_mu.shape} → logits: {_lo.shape}")
del _vae, _x, _lo, _mu, _lv