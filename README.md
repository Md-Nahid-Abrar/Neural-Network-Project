# 🎵 Unsupervised Neural Network for Multi-Genre Music Generation
 
**Member 1:**  Md. Nahid Abrar (24141098)

**Member 2:**  Towsif Khan     (22101186)

.midi files drive link = https://drive.google.com/drive/folders/1oDD36beMtr8Hjmc6wqqoJf1odqIYSnew?usp=sharing 

.ipynb file google colab link = https://colab.research.google.com/drive/1igMZK9JvfhmrRFsNopaL4sWxmhmoLHxT?usp=sharing

 

## 📌 Project Overview
This project explores unsupervised generative modeling for symbolic music (MIDI). By leveraging deep learning architectures, we aim to capture musical structures—such as melody, harmony, and rhythm—without the need for explicit genre labels. The pipeline processes raw MIDI data into piano-roll representations and token sequences to train three distinct generative models.

---

## 📂 Repository Structure
Based on the project environment, the directory is organized as follows:

```text
.
├── dependencies.py
├── EDA.py
├── imports.py
├── loading dataset.py
├── pre-processing.py   
├── Task_Files                              
│        ├──task1/
│             ├── task-1 architecture.py
│             ├── task-1 training.py
│             ├── task-1 results.py             
│             ├── task-1 midi files output.py  
│        ├── task2/
│             ├── task-2 architecture.py
│             ├── task-2 training.py
│             ├── task-2 results.py
│             ├── task-2 midi files output.py
│        ├── task3/
│             ├── task-3 architecture.py
│             ├── task-3 training.py
│             ├── task-3 results.py
│             ├── task-3 midi files output.py
├── maestro-v3.0.0/                       
├── soundfont.sf2                    
├── README.txt

```


## Key Features
**1. Dataset:**

MAESTRO v3.0.0
We utilize the MAESTRO (MIDI and Audio Edited for Synchronous TRacks) dataset.

Size: ~57 MB (MIDI-only version).

Processing: 128-time-step windows with an 88-key piano-roll resolution.

Cleaning: Implementation of a 2% activity threshold to filter out silent musical windows.

**2. Implemented Models**

Task 1: 

LSTM Autoencoder: Learns a compressed 64-dimensional latent representation of 8-second piano-roll segments.

Task 2: 

Variational Autoencoder (VAE): Introduces a probabilistic latent space using the reparameterization trick. Features KL-Annealing and Focal Loss to handle data sparsity.

Task 3: 

Transformer Generator: A decoder-only (GPT-style) architecture using REMI tokenization. Focuses on long-range temporal coherence in musical compositions.

**3. Evaluation Suite**

The models are evaluated against Random and Markov Chain baselines using:

Pitch Histogram Similarity: 

Correlation between generated and ground-truth pitch distributions.

Rhythm Diversity Score:

Variety in quantized note durations.

Repetition Ratio: 

Pitch n-gram analysis to measure musical creativity.

Perplexity (PPL): 
Likelihood-based metric for the Transformer model.

## 🛠️ Setup and Installation
Prerequisites
Ensure you have Python 3.8+ installed. You will also need FluidSynth for MIDI-to-Audio synthesis.

**Linux (Ubuntu/Colab):**

Bash
sudo apt-get install fluidsynth
Python Libraries:

Bash
pip install pretty_midi miditok symusic numpy matplotlib torch torchaudio pyfluidsynth
How to Run
Clone the Repository:

Bash
git clone [https://github.com/your-username/music-generation-nn.git](https://github.com/your-username/music-generation-nn.git)
cd music-generation-nn
Open in Colab/Jupyter: Launch music_generation_project.ipynb.

Download Dataset: 
The first cells in the notebook will automatically download the 57MB MAESTRO dataset.

Train & Generate: 
Execute the cells sequentially to train the models and generate MIDI files in the outputs/ folder.

## 📊 Results
The project provides:

**Loss Curves:** 

Visualizing reconstruction error and KL-divergence.

**Piano-Roll Plots:** 

Comparing original recordings with model-generated outputs.

**Audio Playback:**

Synthesized .wav files using the VintageDreamsWaves soundfont.

