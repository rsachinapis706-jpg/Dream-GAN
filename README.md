<div align="center">

# 🧠 Dream GAN: Making Brainwaves from a Single Sensor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An easy-to-understand AI project that turns simple, single-sensor sleep data into high-quality, full-brain activity maps.*

</div>

---

## 📖 The "Explain It Like I'm 5" Overview

Imagine trying to guess the shape of an entire ocean wave just by watching a single buoy bobbing up and down. That is the problem doctors face when studying sleep at home. 

In a hospital, a patient wears a very uncomfortable cap with **128 sensors (HD-EEG)** to map their entire brain activity while they sleep. But at home, they might only wear **1 simple sensor** on their forehead.

**Dream GAN is an AI that solves this problem.** It learns how the brain's electricity flows. You give it data from just 1 sensor, and it "hallucinates" or accurately predicts what all the other 127 missing sensors would be recording. It turns poor-quality home sleep data into hospital-quality maps!

---

## 🛠️ How It Works (The 2 Phases)

To make this magic happen, the AI uses two main networks that work together in a sequence. 

### 🟢 Phase 1: The Embedding Phase (Learning the Hidden Language)
Before the AI can draw the whole brain, it needs to understand the "hidden language" of the 1 sensor. 
1. The **Embedder Network** looks at the messy, noisy data coming from the single sensor.
2. It compresses this data into a smaller, cleaner "summary" (called a latent space). 
3. This summary captures the *most important* features of the sleep wave, ignoring the noise. 
4. At the same time, a **Recovery Network** learns how to take that summary and turn it back into real brainwave data, ensuring the summary didn't lose anything important.

*Think of Phase 1 like reading a 100-page book and writing a perfect 1-page summary.*

### 🔵 Phase 2: Generative Training (Drawing the Missing Pieces)
Now that the AI understands the summary of the 1 sensor, it is time to generate the missing 127 sensors.
1. The **Generator Network** reads the nice, clean summary created in Phase 1.
2. It uses this summary to "draw" or synthesize what the full 128-sensor brain map should look like.
3. However, the Generator might make mistakes or draw "fake-looking" brainwaves.
4. So, we introduce a **Discriminator Network**. The Discriminator acts like a strict teacher. It looks at *real* 128-sensor data and the Generator's *fake* 128-sensor data, and tries to tell them apart.
5. The Generator keeps practicing until its generated data is so perfect that the Discriminator (the teacher) can no longer tell the difference between the real brainwaves and the AI-generated ones!

*Think of Phase 2 like an art student (Generator) practicing painting while an art critic (Discriminator) keeps telling them what looks fake, until the painting looks completely real.*

---

## 📂 What are all these files?

Don't worry, the project structure is organized nicely so you know where everything is:

```text
Dream-GAN/
│
├── src/                    # The heart of the project (All the Python Code)
│   ├── data/               # Code that loads and cleans the raw brainwave data
│   │   ├── loader.py       # The main boss that hands the data to the AI
│   │   ├── deed_loader.py  # Specific loader for the Donders dataset
│   ├── features/           # Math formulas used to understand the brainwaves
│   ├── models/             # The Actual AI Brains (The Neural Networks)
│   │   ├── timegan.py      # The Generator and Discriminator code
│   │   ├── semantic...     # The Embedder code
│   │   └── losses.py       # The math that calculates how "wrong" the AI's guesses are
│   ├── train.py            # The main script that runs Phase 1 and Phase 2 training
│   ├── train_deed.py       # The training script specifically for the DEED dataset
│   └── visualize_deed.py   # Code that draws pretty pictures of the generated waves
│
├── Q1_Manuscript.tex       # The actual scientific research paper explaining the math
├── presentation.tex        # A slideshow presentation of the project
├── run_project.bat         # A Windows shortcuts - click this to run everything!
└── requirements.txt        # A list of Python tools needed to run this code 
```

---

## 🚀 How to Run the Project (For Beginners)

### Step 1: Install what you need
You will need Python installed on your computer. 
Open your terminal (Command Prompt) and type these commands:

```bash
git clone https://github.com/rsachinapis706-jpg/Dream-GAN.git
cd Dream-GAN

# Install the required Python packages
pip install -r requirements.txt
```

### Step 2: Get the Data
Because brainwave data is huge (gigabytes!), it cannot be stored here. You need to download the **Sleep-EDF** and **Donders (DEED)** datasets and place their folders right inside this main `Dream-GAN` folder.

### Step 3: Start the Magic 🪄
If you are on Windows, simply double-click the `run_project.bat` file! 

If you want to run things manually step-by-step:
```bash
# First, check if your computer has a dedicated graphics card (GPU) 
python check_gpu.py

# Then, start training the AI networks (Phase 1 and Phase 2)
python src/train_unified.py
```

---

## 📊 How do we know it works?

When the AI runs, it draws charts to prove it is learning:
*   **`training_curve.png`**: A line chart showing the AI making fewer and fewer mistakes over time.
*   **`deed_psd_overlap.png`**: A graph showing that the *frequencies* (pitch/speed) of the AI's brainwaves exactly match the frequencies of a real human's brainwaves. 
*   **`tsne_manifold.png`**: A cluster map proving the fake data blends perfectly into the real data.

### 📝 License
This project is open-source. Anyone can learn from it!