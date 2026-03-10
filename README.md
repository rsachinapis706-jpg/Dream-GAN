<div align="center">

# 🧠 Dream GAN: Upgrading Home Sleep Data to Hospital Quality

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An AI project that transforms 1 plain home sleep sensor into 128 high-definition hospital sensors!*
</div>

---

## 📖 The Big Problem 

When you do a sleep study in a hospital, doctors put a **128-sensor cap** on your head. This records exactly how electricity moves across your whole brain. It is very accurate but super uncomfortable to sleep in!

When you do a sleep study at home, you only wear **1 single sensor** on your forehead. It’s comfortable, but doctors miss out on 99% of the information they need to see what your brain is doing.

**Dream GAN is an AI that solves this problem.** 
You give the AI the data from that 1 home sensor, and the AI predicts (or "draws") exactly what all 128 sensors would have recorded if you were wearing the uncomfortable hospital cap. 

---

## 🛠️ How It Works (Explained Simply)

The AI learns how to do this in **two main phases**.

### 🟢 Phase 1: The "Compression/Summary" Phase
Imagine giving an AI a giant, messy 500-page book and asking it to write a clean 1-page summary.

In our project, the AI takes the messy, noisy electricity data from the **1 forehead sensor** and compresses it into a clean, tiny "summary" of the brainwave. 
*   **The Embedder AI:** This creates the tiny summary. It throws away the noise and keeps only the most important pieces of the sleep signal.
*   **The Recovery AI:** This takes the summary and tries to stretch it back out to its original messy state, just to prove that nothing important was accidentally deleted during the summarization.

### 🔵 Phase 2: The "Artist and Judge" Phase (Generative Training)
Now the AI needs to use that clean 1-page summary to "draw" the missing 127 sensors. To do this, we use two AIs that battle each other: the Artist and the Judge.

*   **The Generator AI (The Artist):** The Artist looks at the clean summary from Phase 1. It uses that summary to guess and "paint" the electricity lines for all 128 sensors.
*   **The Discriminator AI (The Judge):** The Judge looks at real 128-sensor hospital data, and then it looks at the Artist's fake 128-sensor data. It tries to spot the fake one and reject it.

At first, the Artist's drawings are terrible, and the Judge rejects them easily. But the Artist keeps practicing over and over again. Eventually, the Artist gets so good at drawing brainwaves that the Judge is completely fooled and thinks the fake data is 100% real. 

**That is when training is complete!** The AI can now accurately predict full-brain activity from just one sensor.

---

## 📂 Project Files Explained

*   **`src/data/`**: The code that loads the raw hospital brainwave files into the computer.
*   **`src/models/`**: Where the actual AI code lives (The Embedder, Recovery, Generator, and Discriminator).
*   **`src/train...`**: The scripts you run to actually start training the AI.
*   **`run_project.bat`**: A simple Windows button that runs all the code automatically.
*   **`Q1_Manuscript.tex`**: Our actual scientific research paper that explains the heavy math.

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
    If you are on Windows, simply double click the `run_project.bat` file to start the whole training process.

---

## 📊 How do we know the AI isn't just tricking us?

After the AI finishes, it creates several charts to prove its "fake" brainwaves are medically accurate:

1.  **`deed_psd_overlap.png`**: This graph proves that the speed and frequency (pitch) of the fake brainwaves match the real ones perfectly.
2.  **`tsne_manifold.png`**: This is a visual map showing that the AIs generated data points perfectly overlap with real human data points.
3.  **`training_curve.png`**: This graph proves the AI made fewer mistakes over time.