# @title EDA — Duration, note-count, pitch & velocity distributions
SAMPLE_N = 100

durations, note_counts, all_pitches, all_velocities = [], [], [], []
midi_paths = [DATA_ROOT / r["midi_filename"] for _, r in meta.iterrows()]
random.shuffle(midi_paths)

for p in midi_paths[:SAMPLE_N]:
    try:
        m   = pretty_midi.PrettyMIDI(str(p))
        ns  = m.instruments[0].notes
        durations.append(m.get_end_time())
        note_counts.append(len(ns))
        all_pitches.extend([n.pitch for n in ns])
        all_velocities.extend([n.velocity for n in ns])
    except Exception:
        pass

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("MAESTRO Dataset — Exploratory Data Analysis", fontsize=15, fontweight="bold")

axes[0, 0].hist(durations, bins=30, color="#4C72B0", edgecolor="white")
axes[0, 0].set(title="Piece Duration (seconds)", xlabel="Duration (s)", ylabel="Count")
axes[0, 0].axvline(np.median(durations), color="red", linestyle="--",
                   label=f"Median: {np.median(durations):.0f}s")
axes[0, 0].legend()

axes[0, 1].hist(note_counts, bins=30, color="#55A868", edgecolor="white")
axes[0, 1].set(title="Notes per Piece", xlabel="Note Count", ylabel="Count")
axes[0, 1].axvline(np.median(note_counts), color="red", linestyle="--",
                   label=f"Median: {int(np.median(note_counts))}")
axes[0, 1].legend()

pitch_counter = Counter(all_pitches)
pitches_sorted = sorted(pitch_counter)
axes[1, 0].bar(pitches_sorted, [pitch_counter[p] for p in pitches_sorted],
               color="#C44E52", width=1.0)
axes[1, 0].set(title="Pitch Distribution (MIDI 0–127)", xlabel="MIDI Pitch", ylabel="Frequency")
axes[1, 0].axvline(60, color="navy", linestyle="--", label="Middle C (60)")
axes[1, 0].legend()

axes[1, 1].hist(all_velocities, bins=40, color="#8172B2", edgecolor="white")
axes[1, 1].set(title="Velocity Distribution", xlabel="Velocity (0–127)", ylabel="Frequency")

plt.tight_layout()
plt.savefig("/content/eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"📊 EDA complete.  Avg duration: {np.mean(durations):.0f}s | "
      f"Avg notes/piece: {int(np.mean(note_counts))}")

# %%
# @title EDA — Piano-roll sparsity check
def piano_roll_sparsity(midi_path, fs=16):
    """Return fraction of active cells in binary piano-roll."""
    try:
        m  = pretty_midi.PrettyMIDI(str(midi_path))
        pr = m.get_piano_roll(fs=fs)[21:109]   # 88 piano keys
        return (pr > 0).sum() / pr.size
    except Exception:
        return None

sparsities = [s for p in midi_paths[:100]
              if (s := piano_roll_sparsity(p)) is not None]

print(f"Mean sparsity (active cells): {np.mean(sparsities)*100:.2f}%")
print(f"→ ~{100-np.mean(sparsities)*100:.1f}% of piano-roll cells are SILENT — "
      "class-imbalance handling is critical!")