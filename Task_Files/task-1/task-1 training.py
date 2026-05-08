"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ### Task 1 — Training
# ──────────────────────────────────────────────────────────────

# %%
# @title Task 1 — Training loop"""
T1_EPOCHS    = 60
T1_LR        = 1e-3
T1_LATENT    = 64
T1_HIDDEN    = 256
T1_LAYERS    = 2
T1_DROPOUT   = 0.3
GRAD_CLIP    = 1.0

ae_model  = LSTMAutoencoder(latent_dim=T1_LATENT, hidden_dim=T1_HIDDEN,
                             num_layers=T1_LAYERS, dropout=T1_DROPOUT).to(DEVICE)
ae_optim  = torch.optim.Adam(ae_model.parameters(), lr=T1_LR)
ae_sched  = torch.optim.lr_scheduler.StepLR(ae_optim, step_size=20, gamma=0.5)
criterion = FocalLoss(alpha=0.75, gamma=2.0)

t1_train_losses, t1_val_losses = [], []

for epoch in range(1, T1_EPOCHS + 1):
    # ---- Train ----
    ae_model.train()
    batch_losses = []
    for x in train_loader:
        x = x.to(DEVICE)
        ae_optim.zero_grad()
        logits, _ = ae_model(x)
        loss      = criterion(logits, x)
        loss.backward()
        nn.utils.clip_grad_norm_(ae_model.parameters(), GRAD_CLIP)
        ae_optim.step()
        batch_losses.append(loss.item())
    t1_train_losses.append(np.mean(batch_losses))

    # ---- Validate ----
    ae_model.eval()
    val_losses = []
    with torch.no_grad():
        for x in val_loader:
            x       = x.to(DEVICE)
            logits, _ = ae_model(x)
            val_losses.append(criterion(logits, x).item())
    t1_val_losses.append(np.mean(val_losses))
    ae_sched.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{T1_EPOCHS}  "
              f"Train Loss: {t1_train_losses[-1]:.4f}  "
              f"Val Loss: {t1_val_losses[-1]:.4f}")

print("\n✅ Task 1 training complete.")