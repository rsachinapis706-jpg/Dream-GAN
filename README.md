<div align="center">

# Dream GAN: Cross-Modal Generative Framework for Sleep EEG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel deep learning framework for zero-shot synthesis of High-Density Electroencephalogram (HD-EEG) signals from sparse, single-channel configurations, utilizing dynamic mode decomposition, sleep microstate topologies, and a cross-modal generative approach.

</div>

---

## 📖 Overview

**Dream GAN (AC-TimeGAN)** addresses a critical challenge in sleep medicine: bridging the gap between clinical accessibility and diagnostic precision. Standard polysomnography often relies on a limited number of EEG channels, restricting the spatial resolution needed to map complex sleep microstates and stage transitions.

By unifying principles from **Dynamic Mode Decomposition (DMD)**, **Topological Data Analysis (TDA)**, and **Time-Series Generative Adversarial Networks (TimeGAN)**, Dream GAN directly synthesizes 128-channel HD-EEG representations from single-channel sleep recordings. The framework facilitates robust, dataset-agnostic sleep stage classification and semantic transition tracking without requiring patient awakenings.

### 🌟 Key Features

*   **Zero-Shot HD-EEG Synthesis**: Reconstructs complete spatial topologies from a single derivation using a localized attention mechanism.
*   **Spectral Preservation**: Inherently preserves critical phase-amplitude couplings and power spectral densities (PSD) observed in physiological sleep spindles and slow waves.
*   **Topological Microstates**: Extracts and encodes quasi-stationary spatio-temporal dynamics into a latent semantic space.
*   **Cross-Modal Embedded Training**: Utilizes an autoencoding architecture and spectral regularizers to prevent mode collapse during unconditional generation.
*   **Generalizability**: Validated across major sleep cohorts including the Donders Sleep Dataset and Sleep-EDF.

---

## 🛠️ System Architecture

The AC-TimeGAN framework is composed of 4 key neural network components trained in three distinct phases (Embedding, Generative training, and Joint Optimization):

1.  **Embedder Network**: Compresses sparse temporal data into a lower-dimensional latent space.
2.  **Generator Network**: Autoregressively synthesizes latent HD-EEG representations conditioned on the embedded input and extracted topological microstates.
3.  **Discriminator Network**: Classifies whether the latent sequences are physiological or synthetic.
4.  **Recovery Network**: Decrypts the temporal representations back into the original 128-channel spatial manifold.

*(See the `Q1_Manuscript.tex` file or generated architectural diagrams for a visual flowchart of operations)*

---

## 📂 Project Structure

```text
Dream-GAN/
│
├── src/                    # Primary Source Code
│   ├── data/               # Data Loaders (Dataset agnostic & specific loaders)
│   │   ├── loader.py       # Core Unified Data Loader Pipeline
│   │   ├── deed_loader.py  # Specific loader handler for Donders/DEED
│   │   └── explore_...     # Scripts for analyzing dataset distributions
│   ├── features/           # Signal processing and feature extraction
│   │   ├── dmd.py          # Dynamic Mode Decomposition logic
│   │   └── microstates.py  # Extraction of quasi-stationary EEG mappings
│   ├── models/             # PyTorch Neural Network Architectures
│   │   ├── timegan.py      # Core TimeGAN formulation
│   │   ├── semantic...     # Semantic and structural regularizers
│   │   └── losses.py       # Custom loss formulations (Reconstruction, Adversarial)
│   ├── train.py            # Main training loop script
│   ├── train_deed.py       # Task-specific training (DEED dataset)
│   └── visualize_deed.py   # Code for plotting generated EEG traces
│
├── Q1_Manuscript.tex       # Complete Academic Manuscript (LaTeX)
├── presentation.tex        # Slide Deck for Project
├── requirements.txt        # Python dependency specifications
├── run_project.bat         # Single-click execution script for Windows
└── .gitignore              # Rules protecting data leaks & caches
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have the following installed:
*   Python 3.8 or higher
*   A CUDA-capable NVIDIA GPU (Highly recommended for the TimeGAN training)
*   Git

### 2. Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/rsachinapis706-jpg/Dream-GAN.git
cd Dream-GAN

# It is highly recommended to use a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Data Setup

Due to size constraints, the raw EEG datasets are **not** included in this repository. 
You will need to download them and place them in the root directory:
*   **Sleep-EDF Database** (Expected folder: `sleep-edf-database-1.0.0/`)
*   **Donders Dream Database (DEED)** (Expected folder: `Dream_Database_Donders/`)

### 4. Running the Project

You can run the entire training pipeline end-to-end using the provided batch script on Windows:

```bash
run_project.bat
```

**Alternatively, step-by-step execution:**

1.  **Check hardware availability:**
    ```bash
    python check_gpu.py
    ```
2.  **Run Spectral Debugging:**
    ```bash
    python src/debug_psd.py
    ```
3.  **Initiate Unified Training Pipeline:**
    ```bash
    python src/train_unified.py
    ```

---

## 📊 Evaluation & Metrics

The framework outputs several artifacts during and after training (found in the root directory or `results/`), which are used directly in the manuscript:

*   **`training_curve.png`**: Visualizes the convergence of the reconstruction and adversarial losses.
*   **`tsne_manifold.png`**: t-SNE plot demonstrating that the synthetic generated distributions perfectly overlap the original physiological data manifold.
*   **`deed_psd_overlap.png`**: Frequency-domain verification showing that generated signals possess the exact Power Spectral Density peaks as real EEG signals (critical for identifying Sleep Spindles).
*   **`retrieval_metrics.png`**: Objective assessment for the quality of generated multivariate data.

---

## 📝 License

This project is open-source. Please see the [LICENSE](LICENSE) file for more details. If you utilize portions of this codebase for academic research, kindly cite the `Q1_Manuscript.tex` appropriately.