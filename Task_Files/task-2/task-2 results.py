"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ### Task 2 — Results & Loss Curve
# ──────────────────────────────────────────────────────────────

# %%
# @title Task 2 — Plot VAE loss curves"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(t2_train_total, label="Train Total", color="#4C72B0", linewidth=2)
axes[0].plot(t2_val_total,   label="Val Total",   color="#C44E52",
             linewidth=2, linestyle="--")
axes[0].set(title="Task 2 — VAE: Total Loss", xlabel="Epoch", ylabel="Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(t2_train_recon, label="Recon Loss",   color="#55A868", linewidth=2)
axes[1].plot(t2_train_kl,    label="KL Divergence",color="#8172B2", linewidth=2,
             linestyle="--")
axes[1].axvline(KL_WARMUP, color="red", linestyle=":", alpha=0.7,
                label=f"KL warmup ends (ep {KL_WARMUP})")
axes[1].set(title="Task 2 — VAE: Recon vs KL Loss", xlabel="Epoch", ylabel="Loss")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/content/task2_loss_curves.png", dpi=150)
plt.show()