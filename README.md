# Neural-Network-Project
Course: CSE425 / EEE474 — Neural Networks

Target: Unsupervised representation learning for symbolic music (MIDI).

This repository contains an end-to-end pipeline for generating multi-genre music using various deep learning architectures. Unlike supervised models, these networks learn musical structures, rhythms, and pitch distributions without explicit genre labels.

📂 Repository Structure
Based on the project environment, the directory is organized as follows:

Plaintext
.
├── music_generation_project.ipynb   # Main Google Colab / Jupyter Notebook
├── outputs/                         # Generated MIDI and Audio files
│   ├── task1/                       # LSTM Autoencoder samples
│   ├── task2/                       # VAE samples & latent interpolations
│   ├── task3/                       # Transformer compositions
│   └── baselines/                   # Random & Markov Chain samples
├── maestro-v3.0.0/                  # Dataset directory (MIDI-only, ~57 MB)
├── soundfont.sf2                    # FluidSynth soundfont for MIDI synthesis
└── README.md                        # Project documentation
🚀 Project Overview
1. Dataset
The model is trained on the MAESTRO Dataset (v3.0.0), a lightweight (57 MB) collection of high-quality MIDI recordings.

Pre-processing: Segments music into 128-step piano-roll windows (88 keys).

Data Cleaning: Filters out sparse sequences (less than 2% active notes) to ensure the models learn meaningful patterns.

2. Core Tasks
The project implements three distinct generative approaches:

Task 1: LSTM Autoencoder (Reconstruction)

Uses an LSTM bottleneck to compress 8-second windows into a 64-dimensional latent vector.

Focuses on learning the fundamental structural reconstruction of piano-rolls.

Task 2: Variational Autoencoder (VAE) (Diversity)

Implements a Gaussian latent space with the reparameterization trick.

Uses KL-Annealing and Focal Loss to manage the data's inherent sparsity.

Includes Latent Space Interpolation to smoothly transition between musical styles.

Task 3: Transformer Generator (Coherence)

A decoder-only (GPT-style) Transformer using REMI tokenization via miditok.

Implements causal masking and top-k sampling for long-form, coherent compositions.

3. Evaluation Metrics
Models are compared against Random and Markov Chain baselines using:

Pitch Histogram Similarity: Comparison of generated vs. real pitch distributions.

Rhythm Diversity: Analysis of quantized note-duration variety.

Repetition Ratio: Pitch n-gram analysis to detect creativity vs. mechanical repetition.

Perplexity (PPL): Likelihood-based evaluation for the Transformer model.

🛠️ Installation & Usage
Prerequisites
The project requires FluidSynth for MIDI synthesis and several Python libraries:

Bash
# System dependency (for audio playback)
sudo apt-get install fluidsynth

# Python dependencies
pip install pretty_midi miditok torch numpy matplotlib pyfluidsynth
Running the Project
Open music_generation_project.ipynb in Google Colab.

Run Section 1 & 2 to install dependencies and download the MAESTRO dataset.

Execute the training blocks for Tasks 1, 2, or 3.

Check the /outputs/ folder for generated MIDI files and listen to the synthesized WAV results directly in the notebook.

📊 Results & Visualizations
The project generates several analytical plots, including:

Training Curves: Reconstruction Loss (BCE) and KL-Divergence.

Piano-Rolls: Visual comparisons of original vs. reconstructed vs. generated music.

Metric Tables: Comparative analysis of all models against the baselines.

📜 License
This project was developed for the CSE425/EEE474 Neural Networks course. All code and methodologies are intended for educational and research purposes.

Author: [Your Name/GitHub Handle]

Date: April 2026
