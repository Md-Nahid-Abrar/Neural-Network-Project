"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ### Task 1 — Output & Playing 1 Generated MIDI File
# ──────────────────────────────────────────────────────────────

# %%
# @title Task 1 — Generate 5 MIDI samples"""
os.makedirs("/content/task1_outputs", exist_ok=True)

ae_model.eval()
t1_midi_paths = []

with torch.no_grad():
    for i in range(5):
        z      = torch.randn(1, T1_LATENT).to(DEVICE)
        logits = ae_model.decoder(z)                    # (1, T, 88)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()   # (T, 88)
        path   = f"/content/task1_outputs/t1_sample_{i+1}.mid"
        piano_roll_to_midi(probs, threshold=0.35, save_path=path)
        t1_midi_paths.append(path)
        print(f"  ✅ Generated: {path}")

print(f"\n🎵 5 MIDI samples saved to /content/task1_outputs/")

# %%
# @title Task 1 — Play Sample #1
print("🎵 Playing Task 1 — LSTM Autoencoder, Sample 1")
play_midi(t1_midi_paths[0])
