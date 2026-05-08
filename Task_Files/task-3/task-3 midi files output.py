"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ### Task 2 — Output & Playing 1 Generated MIDI File
# ──────────────────────────────────────────────────────────────

# %%
# @title Task 2 — Generate 8 MIDI samples (multi-genre via latent diversity)"""
os.makedirs("/content/task2_outputs", exist_ok=True)

vae_model.eval()
t2_midi_paths = []

with torch.no_grad():
    probs_batch = vae_model.sample(n=8).cpu().numpy()   # (8, T, 88)
    for i, probs in enumerate(probs_batch):
        path = f"/content/task2_outputs/t2_sample_{i+1}.mid"
        piano_roll_to_midi(probs, threshold=0.35, save_path=path)
        t2_midi_paths.append(path)
        print(f"  ✅ Generated: {path}")

print(f"\n🎵 8 MIDI samples saved to /content/task2_outputs/")

# %%
# @title Task 2 — Latent Interpolation Experiment
# Encode two real pieces → interpolate μ₁ → μ₂ → decode 8 steps
os.makedirs("/content/task2_interpolation", exist_ok=True)

real_batch = next(iter(val_loader))[:2].to(DEVICE)   # 2 real pieces

vae_model.eval()
with torch.no_grad():
    mu1, _ = vae_model.encoder(real_batch[0:1])
    mu2, _ = vae_model.encoder(real_batch[1:2])

    interp_paths = []
    for i, alpha in enumerate(np.linspace(0, 1, 8)):
        z_alpha = (1 - alpha) * mu1 + alpha * mu2
        probs   = torch.sigmoid(vae_model.decoder(z_alpha)).squeeze(0).cpu().numpy()
        path    = f"/content/task2_interpolation/interp_{i+1}.mid"
        piano_roll_to_midi(probs, threshold=0.35, save_path=path)
        interp_paths.append(path)

print(f"✅ Latent interpolation: 8 MIDI files saved to /content/task2_interpolation/")

# %%
# @title Task 2 — Play Sample #1
print("🎵 Playing Task 2 — VAE, Sample 1")
play_midi(t2_midi_paths[0])