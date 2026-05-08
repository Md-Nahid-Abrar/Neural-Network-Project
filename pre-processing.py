"""# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ## ⚙️ Section 5 — Preprocessing (Piano-Roll Windows)
# ──────────────────────────────────────────────────────────────

# %%
# @title Preprocessing — build piano-roll windows for Tasks 1 & 2"""
FS          = 16      # frames per second
WIN_LEN     = 128     # time steps per window  (= 8 s at 16 fps)
MIN_DENSITY = 0.02    # discard windows with < 2 % active cells
MAX_FILES   = 300     # limit files to keep Colab memory manageable

def midi_to_windows(path, fs=FS, win_len=WIN_LEN, min_density=MIN_DENSITY):
    """Return list of binary piano-roll windows of shape (win_len, 88)."""
    try:
        m  = pretty_midi.PrettyMIDI(str(path))
        pr = m.get_piano_roll(fs=fs)[21:109].T   # (T, 88)
        pr = (pr > 0).astype(np.float32)          # binarise
        windows = []
        for start in range(0, len(pr) - win_len + 1, win_len):
            w = pr[start : start + win_len]
            if w.mean() >= min_density:
                windows.append(w)
        return windows
    except Exception:
        return []

# Build dataset — takes ~2–3 min for 300 files on Colab
train_meta = meta[meta["split"] == "train"].head(MAX_FILES)
val_meta   = meta[meta["split"] == "validation"].head(60)
test_meta  = meta[meta["split"] == "test"].head(40)

def build_windows(subset_meta):
    all_w = []
    for _, row in subset_meta.iterrows():
        all_w.extend(midi_to_windows(DATA_ROOT / row["midi_filename"]))
    return np.stack(all_w).astype(np.float32) if all_w else np.empty((0, WIN_LEN, 88))

print("⏳ Building training windows …")
X_train = build_windows(train_meta)
print("⏳ Building validation windows …")
X_val   = build_windows(val_meta)
print("⏳ Building test windows …")
X_test  = build_windows(test_meta)

print(f"\n✅ Train windows : {X_train.shape}")
print(f"   Val windows   : {X_val.shape}")
print(f"   Test windows  : {X_test.shape}")
print(f"   Memory usage  : {X_train.nbytes / 1e6:.1f} MB (train)")

# %%
# @title PyTorch Dataset & DataLoaders for piano-roll tasks
class PianoRollDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)   # (N, 128, 88)
    def __len__(self):  return len(self.data)
    def __getitem__(self, i):  return self.data[i]

BATCH_SIZE = 64
train_loader = DataLoader(PianoRollDataset(X_train), batch_size=BATCH_SIZE,
                          shuffle=True,  drop_last=True)
val_loader   = DataLoader(PianoRollDataset(X_val),   batch_size=BATCH_SIZE,
                          shuffle=False, drop_last=False)
test_loader  = DataLoader(PianoRollDataset(X_test),  batch_size=BATCH_SIZE,
                          shuffle=False, drop_last=False)
print(f"✅ DataLoaders ready.  Train batches: {len(train_loader)}")

# ──────────────────────────────────────────────────────────────
# %% [markdown]
# ## 🏗️ Section 6 — Shared Utilities (Loss · MIDI Export · Playback)
# ──────────────────────────────────────────────────────────────

# %%
# @title Focal Loss (handles severe piano-roll class imbalance)
class FocalLoss(nn.Module):
    """Binary Focal Loss: down-weights easy negatives (silent cells)."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt   = torch.where(targets == 1, probs, 1 - probs)
        at   = torch.where(targets == 1,
                           torch.tensor(self.alpha, device=logits.device),
                           torch.tensor(1 - self.alpha, device=logits.device))
        loss = at * (1 - pt) ** self.gamma * bce
        return loss.mean()

# %%
# @title Piano-roll → MIDI export helper
SOUNDFONT = "/content/soundfont.sf2"

def piano_roll_to_midi(roll_np, fs=FS, threshold=0.35,
                       velocity=80, program=0, save_path=None):
    """
    Convert a binary/probability (T, 88) array to a MIDI file.
    threshold < 0.5 because models trained on imbalanced data
    tend to underestimate note probabilities.
    """
    if save_path is None:
        save_path = "/content/generated.mid"
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, name="Piano")
    roll = (roll_np >= threshold).astype(np.uint8)
    for pitch_idx in range(88):
        midi_pitch = pitch_idx + 21
        active = False
        onset  = 0.0
        for t in range(len(roll)):
            if roll[t, pitch_idx] and not active:
                onset  = t / fs
                active = True
            elif not roll[t, pitch_idx] and active:
                offset = t / fs
                dur    = offset - onset
                if dur >= 1 / fs:        # ignore zero-length notes
                    inst.notes.append(
                        pretty_midi.Note(velocity=velocity,
                                         pitch=midi_pitch,
                                         start=onset, end=offset))
                active = False
        if active:
            inst.notes.append(
                pretty_midi.Note(velocity=velocity, pitch=midi_pitch,
                                 start=onset, end=len(roll) / fs))
    pm.instruments.append(inst)
    pm.write(save_path)
    return save_path

def midi_to_audio(midi_path, sf_path=SOUNDFONT, out_path=None):
    """Synthesise MIDI → WAV with FluidSynth; return IPython Audio widget."""
    if out_path is None:
        out_path = midi_path.replace(".mid", ".wav")
    try:
        subprocess.run(
            ["fluidsynth", "-ni", sf_path, midi_path, "-F", out_path, "-r", "22050"],
            capture_output=True, timeout=30)
        return Audio(out_path, autoplay=False)
    except Exception as e:
        print(f"⚠️  Audio synthesis failed: {e}\n"
              "    Download the MIDI file and use MuseScore / GarageBand for playback.")
        return None

def play_midi(midi_path):
    """Display MIDI playback widget in Colab."""
    widget = midi_to_audio(midi_path)
    if widget:
        display(widget)
    display(HTML(f'<b>MIDI file saved:</b> <code>{midi_path}</code>'))