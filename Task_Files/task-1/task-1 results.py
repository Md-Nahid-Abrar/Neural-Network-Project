
"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ### Task 1 — Results & Loss Curve
# ──────────────────────────────────────────────────────────────

# %%
# @title Task 1 — Plot reconstruction loss curve"""
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t1_train_losses, label="Train Loss", color="#4C72B0", linewidth=2)
ax.plot(t1_val_losses,   label="Val Loss",   color="#C44E52", linewidth=2,
        linestyle="--")
ax.set(title="Task 1 — LSTM Autoencoder: Focal Loss Curve",
       xlabel="Epoch", ylabel="Focal Loss")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/content/task1_loss_curve.png", dpi=150)
plt.show()
print(f"Best Val Loss: {min(t1_val_losses):.4f} at epoch "
      f"{np.argmin(t1_val_losses)+1}")