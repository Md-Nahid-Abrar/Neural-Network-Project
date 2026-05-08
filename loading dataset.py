DATA_ROOT   = Path("/content/maestro")
MIDI_ROOT   = DATA_ROOT                         # MIDIs live directly under year folders
CSV_PATH    = DATA_ROOT / "maestro-v3.0.0.csv"

if not CSV_PATH.exists():
    URL = ("https://storage.googleapis.com/magentadata/datasets/"
           "maestro/v3.0.0/maestro-v3.0.0-midi.zip")
    zip_path = "/content/maestro-midi.zip"
    print("⬇️  Downloading MAESTRO MIDI-only (~57 MB) …")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/content/")
    print("✅ Dataset extracted.")
else:
    print("✅ Dataset already present.")

meta = pd.read_csv(CSV_PATH)
print(f"   Total recordings: {len(meta)}")
print(meta["split"].value_counts().to_string())

# %%
# @title Verify a sample MIDI file loads correctly
sample_row  = meta[meta["split"] == "train"].iloc[0]
sample_path = DATA_ROOT / sample_row["midi_filename"]
pm          = pretty_midi.PrettyMIDI(str(sample_path))
notes       = pm.instruments[0].notes
print(f"Sample file : {sample_path.name}")
print(f"Duration    : {pm.get_end_time():.1f} s")
print(f"Note count  : {len(notes)}")
print(f"First note  : pitch={notes[0].pitch}, vel={notes[0].velocity}, "
      f"start={notes[0].start:.3f}s, end={notes[0].end:.3f}s")