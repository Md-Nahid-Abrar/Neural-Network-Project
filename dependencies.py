!pip install pretty_midi miditok symusic -q
!pip install matplotlib numpy pandas torch torchaudio -q
!apt-get install -y fluidsynth -q          # MIDI → WAV synthesis
!pip install pyfluidsynth -q               # Python bindings for FluidSynth
!wget -q "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2" \
     -O /content/soundfont.sf2 2>/dev/null || \
  wget -q "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip" -O /tmp/sf.zip && \
  unzip -q /tmp/sf.zip -d /content && mv /content/FluidR3_GM.sf2 /content/soundfont.sf2 || true
# Fallback soundfont download
!wget -q "https://musical-artifacts.com/artifacts/1461/TimGM6mb.sf2" \
     -O /content/soundfont.sf2 2>/dev/null || true

print("✅ All libraries installed.")