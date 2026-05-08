"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ### Task 2 — Training with KL Annealing
# ──────────────────────────────────────────────────────────────

# %%
# @title Task 2 — VAE training loop with KL annealing"""
T2_EPOCHS    = 70
T2_LR        = 1e-3
T2_LATENT    = 64
T2_HIDDEN    = 256
T2_LAYERS    = 2
T2_DROPOUT   = 0.3
KL_WARMUP    = 30          # epochs before β starts rising
KL_MAX       = 1.0

vae_model = MusicVAE(latent_dim=T2_LATENT, hidden_dim=T2_HIDDEN,
                     num_layers=T2_LAYERS, dropout=T2_DROPOUT).to(DEVICE)
vae_optim = torch.optim.Adam(vae_model.parameters(), lr=T2_LR)
vae_sched = torch.optim.lr_scheduler.StepLR(vae_optim, step_size=25, gamma=0.5)

t2_train_total, t2_val_total = [], []
t2_train_recon, t2_train_kl  = [], []

for epoch in range(1, T2_EPOCHS + 1):
    # KL annealing: β linearly ramps from 0 → KL_MAX over KL_WARMUP epochs
    beta = min(KL_MAX, KL_MAX * max(0, epoch - KL_WARMUP) / KL_WARMUP)

    # ---- Train ----
    vae_model.train()
    e_total, e_recon, e_kl = [], [], []
    for x in train_loader:
        x = x.to(DEVICE)
        vae_optim.zero_grad()
        logits, mu, lv = vae_model(x)
        loss, r, k     = vae_loss(logits, x, mu, lv, beta=beta)
        loss.backward()
        nn.utils.clip_grad_norm_(vae_model.parameters(), GRAD_CLIP)
        vae_optim.step()
        e_total.append(loss.item()); e_recon.append(r.item()); e_kl.append(k.item())
    t2_train_total.append(np.mean(e_total))
    t2_train_recon.append(np.mean(e_recon))
    t2_train_kl.append(np.mean(e_kl))

    # ---- Validate ----
    vae_model.eval()
    v_total = []
    with torch.no_grad():
        for x in val_loader:
            x = x.to(DEVICE)
            logits, mu, lv = vae_model(x)
            loss, _, _ = vae_loss(logits, x, mu, lv, beta=beta)
            v_total.append(loss.item())
    t2_val_total.append(np.mean(v_total))
    vae_sched.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{T2_EPOCHS}  "
              f"Total: {t2_train_total[-1]:.4f}  "
              f"Recon: {t2_train_recon[-1]:.4f}  "
              f"KL: {t2_train_kl[-1]:.4f}  β={beta:.2f}")

print("\n✅ Task 2 training complete.")