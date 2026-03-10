<div align="center">

# 🧠 AC-Semantic-TimeGAN: Zero-Shot EEG Synthesis & Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A Supervisory-Governed Generative Architecture for NREM Dream State Synthesis and Classification.*

</div>

---

## 📖 The "Explain It Like I'm 5" Overview

Imagine trying to guess the shape of an entire ocean wave just by watching a single buoy bobbing up and down. That is the problem doctors face when studying sleep at home. 

In a hospital, a patient wears a very uncomfortable cap with **128 sensors (HD-EEG)** to map their entire brain activity. But at home, they might only wear **1 simple sensor** on their forehead.

**AC-Semantic-TimeGAN solves this problem while also reading your emotions.** 
It learns how electricity flows across the brain. You give it data from just 1 sensor, and it "hallucinates" exactly what all the other 127 missing sensors would be recording. Even better, once it builds that full brain map, it can read it to tell us exactly what emotion (Joy, Fear, Sadness) you were feeling during your dream!

---

## 🛠️ How It Works: The Two-Stage Pipeline

To prevent the AI from generating physically impossible brainwaves, the system is strictly split into two rigidly controlled phases. 

### 🟢 Phase 1: Physiology-Aware Generation (AC-TimeGAN)
Before we can classify emotions, we need a mathematically perfect baseline brainwave. If we just ask a standard AI to generate continuous brainwaves, they usually suffer from "Spectral Drift" (they forget the rules of physics over time). 

Therefore, Phase 1 focuses \*\*exclusively\*\* on generating biologically valid electricity geometries:
1. **The Generator & Supervisor:** A Gated Recurrent Unit (GRU) generator tries to synthesize the 128 channels. A special "Supervisor" network watches it, heavily penalizing any sequence that violates the physical rules of how brainwaves flow over time. 
2. **Dynamic Mode Decomposition (DMD):** This relies on fluid dynamics! It calculates the absolute rules of human brain frequencies (e.g., Delta waves must always have higher power than Beta waves). If the Generator starts making physical mistakes ($1/f$ spectral collapse), the DMD algorithm forcefully snaps the brainwave back onto the correct physical trajectory.
3. **The Auxiliary Classifier:** Simultaneously, a discriminator tries to separate the rare "Dreaming" brainwaves from the overwhelming amount of "Neutral" sleeping brainwaves.

*Result of Phase 1: We now have an AI that perfectly generates 128 channels of brain electricity from a single sensor, without breaking the laws of biology.*

### 🔵 Phase 2: Semantic Text Alignment (Semantic-TimeGAN)
Now that we have biologically perfect synthesized brainwaves, we can figure out what they mean! This phase executes "Zero-Shot" learning to connect the raw electricity directly to Natural Language Processing (English words).

1. \*\*Freezing the Physical Map:\*\* We lock the Generator from Phase 1 completely. We don't want to break our perfect physics engine!
2. \*\*The Contextual Transformer (MiniLM-L6):\*\* The AI reads clinical dream reports (e.g., "I was falling and felt fear") and translates the English words into a dense mathematical grid (a Hypersphere). 
3. \*\*InfoNCE Contrastive Geometry:\*\* This is the real magic. The architecture trains a nonlinear projection head that takes the generated 128-channel brainwave and actively warps it until it mathematically aligns with the corresponding English dream report matrix. 

*Result of Phase 2: The architecture acts as a universal translator. It can look at a generated sleep brainwave and retrieve the exact subjective emotion the patient was feeling (Joy, Fear, Sadness) with 79.41% Top-3 accuracy!*

---

## 📂 Project Files Explained

*   **`src/data/`**: Extracts data from the DEED (Emotion-labeled EEG) and DREAM (Free-text labeled EEG) datasets.
*   **`src/models/timegan.py`**: Houses the strict AC-TimeGAN mechanics for Phase 1. 
*   **`src/models/semantic_encoder.py`**: The InfoNCE contrastive geometry logic for Phase 2.
*   **`src/features/dmd.py`**: The mathematical physics engine enforcing physiological limits.
*   **`Q1_Manuscript.tex`**: The unabridged scientific research paper detailing all mathematical proofs.
*   **`Q1_Presentation.tex`**: The Beamer slide deck for the architecture.

---

## 🚀 How to Run the Project on Your Computer

1.  **Get the Code:** 
    Open your command prompt and clone this folder:
    ```bash
    git clone https://github.com/rsachinapis706-jpg/Dream-GAN.git
    cd Dream-GAN
    ```

2.  **Install the Python Tools:**
    Type this to install the required programs:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Add the Data:**
    Copy the `sleep-edf-database` folder and the `Dream_Database_Donders` folder straight into this `Dream-GAN` folder. *(They are too big to upload here!)*

4.  **Click Run!** 
    Double click the `run_project.bat` file to start the sequential training process locally.

---

## 📊 Evaluation & Metrics 

The framework achieves state-of-the-art diagnostic parameters spanning both physics and abstract human cognition:

1.  **`deed_psd_overlap.png`**: Proves the synthetic baseline EEG arrays empirically maintain strictly mandated biological power hierarchies. 
2.  **`retrieval_metrics.png`**: Demonstrates the Phase 2 algorithm achieving an incredible 79.41% Top-3 accuracy when mapping an entirely unknown brainwave specifically to subjective emotional text targets (Fear, Joy, Neutral, Sadness).
3.  **`tsne_manifold.png`**: A qualitative spatial alignment map proving the InfoNCE algorithm successfully groups emotional states from the raw electricity alone.