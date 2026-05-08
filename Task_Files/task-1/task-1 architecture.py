"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ## 🧩 Task 1 — LSTM Autoencoder (Single-Genre)
# ### Architecture
# ──────────────────────────────────────────────────────────────

# %%
# @title Task 1 — LSTM Autoencoder architecture"""
class LSTMEncoder(nn.Module):
    """
    Encoder: (T=128, 88) → z ∈ R^latent_dim
    Uses final hidden state of last LSTM layer as bottleneck.
    """
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, T, 88)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, B, hidden_dim) — take last layer
        z = self.fc(h_n[-1])             # (B, latent_dim)
        return z


class LSTMDecoder(nn.Module):
    """
    Decoder: z ∈ R^latent_dim → (T=128, 88) logits
    z is repeated across all time steps AND concatenated to LSTM input
    so the latent code is visible throughout decoding.
    No activation at output — raw logits for numerical stability.
    """
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=88,
                 seq_len=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.seq_len    = seq_len
        self.latent_dim = latent_dim
        self.fc_in      = nn.Linear(latent_dim, latent_dim)  # project z
        self.lstm       = nn.LSTM(latent_dim, hidden_dim, num_layers=num_layers,
                                  batch_first=True, dropout=dropout)
        self.fc_out     = nn.Linear(hidden_dim, output_dim)  # raw logits

    def forward(self, z):
        # z: (B, latent_dim)
        z_proj = self.fc_in(z).unsqueeze(1)                  # (B, 1, latent_dim)
        z_rep  = z_proj.repeat(1, self.seq_len, 1)            # (B, T, latent_dim)
        out, _ = self.lstm(z_rep)                             # (B, T, hidden_dim)
        logits = self.fc_out(out)                             # (B, T, 88)
        return logits


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64,
                 seq_len=WIN_LEN, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim,
                                   num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim,
                                   seq_len,   num_layers, dropout)

    def forward(self, x):
        z      = self.encoder(x)
        logits = self.decoder(z)
        return logits, z

print("✅ Task 1 architecture defined.")
# Quick sanity check
_model = LSTMAutoencoder().to(DEVICE)
_dummy = torch.zeros(4, WIN_LEN, 88).to(DEVICE)
_out, _z = _model(_dummy)
print(f"   Input:  {_dummy.shape}  →  z: {_z.shape}  →  Output: {_out.shape}")
del _model, _dummy, _out, _z