# AC-Semantic-TimeGAN: From Physiological Fidelity to Semantic Alignment in Generative Dream EEG Modeling

## Abstract
Electroencephalogram (EEG)-based dream modeling remains challenging due to the need for both physiological realism and semantic interpretability. Existing generative approaches often prioritize signal fidelity while lacking mechanisms to relate neural activity to meaningful cognitive states. In this work, we propose AC-Semantic-TimeGAN, a two-stage generative framework. First, AC-TimeGAN introduces physiology-aware constraints to preserve spectral structure, microstate dynamics, and downstream classification robustness in synthetic dream EEG generation. Building upon this foundation, Semantic-TimeGAN extends the model with a cross-modal semantic alignment module that enables text-conditioned EEG synthesis and zero-shot retrieval between EEG signals and affective language representations. Experiments on sleep EEG datasets and a multimodal EEG-text dream dataset demonstrate high physiological fidelity, improved classification accuracy (78.01%), and statistically significant semantic alignment (46% Top-1 and 76% Top-3 zero-shot retrieval accuracy with mean cosine similarity of 0.92). These results suggest that physiological validity and semantic interpretability can be jointly achieved in generative EEG modeling without supervised emotion decoding.

## 1. Introduction

Decoding the latent cognitive variables embedded within human sleep states represents a critical threshold in brain-computer interface (BCI) research \cite{amrita2022eeg}. Traditional sleep analysis relies heavily on the visual scoring of polysomnograms, compartmentalizing continuous neural activity into discrete, macro-level physiological states (e.g., Non-Rapid Eye Movement [NREM] stages N1-N3 and Rapid Eye Movement [REM] sleep) \cite{anitha2024efficient}. However, this macroscopic staging inherently filters out the high-dimensional spatial and temporal variability intrinsic to internal cognitive phenomena, such as dreaming and parasomnia-related imagery \cite{10.3389/fnhum.2019.00348}. Specifically, investigating the "Cortical Hot Zone"—the posterior parahippocampal and occipital regions strongly correlated with dream onset—requires continuous, high-fidelity signal analysis rather than discrete epoch staging \cite{siclari2018dreaming}. 

The primary barrier to advancing machine learning applications in this domain is severe data scarcity. High-density EEG recordings (e.g., 64 to 256 channels) matched with continuous, time-labeled cognitive reports are logistically prohibitive to acquire. Clinical recording environments are highly susceptible to motor artifacts, electrode degradation over multi-hour sessions, and the subjective recall bias of patients upon waking. Consequently, generative adversarial networks (GANs) and other latent generative models have been proposed as a mathematical mechanism to artificially augment clinical EEG datasets. 

Despite their success in computer vision, conventional time-series GANs fail when applied to neurophysiological data due to an inability to enforce inherent biological constraints. Standard adversarial discriminators rely solely on statistical distribution matching, ignoring the well-documented spectral $1/f$ aperiodic component and cross-frequency phase coupling intrinsic to human cortical activity. When conventional GANs generate raw voltage potentials, they frequently induce catastrophic spectral distortion, where the generated high-frequency beta and gamma bands collapse into Gaussian noise, ultimately destroying the transient microstate topographies required for actual clinical or cognitive analysis. 

Conversely, models configured exclusively for EEG emotion tracking and semantic decoding operate strictly as dense discriminators or supervised classifiers. While capable of extracting features from real signals, these classifiers lack the probabilistic mapping mechanisms required to invert the process and conditionally synthesize raw neural data from affective states. This disconnect isolates the physiological synthesis problem from the semantic interpretation problem.

To overcome the mutual exclusivity of signal realism and semantic interpretability, we propose a two-stage framework that first ensures biological plausibility and subsequently introduces semantic alignment without compromising initial structural validation. 

Specifically, the core contributions of this work are fourfold:
1. We introduce AC-TimeGAN, a physiology-aware generative framework that preserves spectral hierarchy, microstate syntax, and classification robustness in synthetic dream EEG signals via specific structural regularization.
2. We propose Semantic-TimeGAN, a novel extension incorporating a self-supervised, cross-modal projection head that enables strict text-conditioned EEG generation and zero-shot semantic retrieval.
3. We demonstrate that robust physiological fidelity and statistically significant semantic alignment can be extracted simultaneously without relying on fully-supervised, epoch-by-epoch emotion decoding constraints.
4. We provide transparent, extensive quantitative and qualitative evaluations covering transition entropy, frequency-band preservation, and topological manifold mapping across distinct clinical and cognitive datasets.

## 2. Related Work

### 2.1 Latent Generation of Physiological Time-Series

The adaptation of Generative Adversarial Networks to continuous time-series modalities has accelerated rapidly, largely driven by architectures like TimeGAN \cite{yoon2019time}, which introduced a joint temporal and unsupervised loss framework utilizing RNN-based latent dynamics. Variations such as GAN-EEG extended this concept to multi-channel biosignals, primarily aiming to augment motor-imagery BCIs or epileptic seizure databases \cite{amrita2025single}. 

However, neurobiological validation frameworks applied to these generated signals reveal severe structural deficits. Zuo et al. \cite{zuo2025high} demonstrated that recurrent generators optimized purely via adversarial loss consistently fail to reproduce the non-stationary, oscillatory burst phenomena—such as K-complexes and occipital sleep spindles—critical for sleep phase transition analysis \cite{diezig2024eeg}. Furthermore, while statistical features in the time domain may appear visually coherent, frequency-domain transformations applied to classical GAN-EEG outputs frequently exhibit "spectral flattening," breaking the physiological power laws governing cortical dynamics. A generative model that cannot reproduce the transient microstate transition matrices or the structural frequency ratios of real brainwaves remains fundamentally unusable for downstream clinical modeling \cite{10.3389/fnins.2023.1219133}.

### 2.2 Emotion Classification and Semantic Alignment

Parallel to generation, the domain of affective computing has focused heavily on extracting emotional valence and semantic context from EEG variants. Current methodologies apply weakly supervised learning, spatial-temporal graph convolutional networks (ST-GCN), and Transformer-based decoders to classify baseline states inside highly controlled stimulus environments (e.g., viewing affective video clips) \cite{naidu2024emotion}.

While these analytical models excel at identifying localized frequency shifts associated with arousal or valence, they terminate in finite discrete classifications. They do not maintain a generative bidirectional mapping between the text-based semantic space and the raw EEG manifold. Translating continuous free-form affective language (such as dream reports) directly into multidimensional physiological brainwave synthesis has remained entirely unaddressed by current architectures. 

To the best of our knowledge, no prior work jointly addresses physiological EEG fidelity and semantic alignment within a single progressive generative framework. Existing models either generate artifact-heavy raw data oblivious to meaning or classify meaning from biological data without the capacity to synthesize it.

## 3. System Architecture: The AC-Semantic-TimeGAN Framework

The proposed framework operates as a sequentially constrained structural pipeline. Rather than enforcing multi-modal alignment and physiological synthesis simultaneously—which empirically guarantees topological collapse—we decouple the objective into two mathematically distinct phases. Phase 1 (AC-TimeGAN) establishes a robust, biologically accurate generative manifold. Phase 2 (Semantic-TimeGAN) maps natural language embeddings onto this pre-stabilized manifold using a cross-modal projection head. Semantic-TimeGAN operates on the physiologically validated output of AC-TimeGAN and does not alter the underlying biological constraints.

### 3.1 Phase 1: Physiology-Aware AC-TimeGAN

To synthesize high-fidelity polysomnographic signals, AC-TimeGAN expands upon traditional recurrent adversarial networks by integrating an Auxiliary Classifier and explicit structural boundary loss functions. Let $\mathcal{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$ represent the multivariate temporal EEG dataset, where $\mathbf{x}_i \in \mathbb{R}^{T \times C}$, with $T$ denoting sequence length and $C$ representing the spatial sensor channels.

The architecture comprises four primary nonlinear mapping functions parameterized by deep Gated Recurrent Unit (GRU) networks: an embedding temporal network $e: \mathcal{X} \rightarrow \mathcal{H}$, a recovery network $r: \mathcal{H} \rightarrow \hat{\mathcal{X}}$, a sequence generator $g: \mathcal{Z} \rightarrow \hat{\mathcal{H}}$, and an adversarial discriminator $d: \mathcal{H} \rightarrow [0,1]$. 

To circumvent the spectral decay ubiquitous in standard adversarial generation, AC-TimeGAN enforces Physiology-Aware Alignment (PAA) directly across the temporal latent space $\mathcal{H}$. This is achieved by penalizing deviations from the inverse-frequency ($1/f$) scaling laws governing true local field potentials:
$$ \mathcal{L}_{PAA} = \sum_{c=1}^{C} \left\| \log(S_{\mathbf{x}}(f)) - \log(S_{\hat{\mathbf{x}}}(f)) \right\|_2^2 $$
where $S(f)$ denotes the power spectral density computation via the continuous wavelet transform.

Simultaneously, the Auxiliary Classifier network assesses the generated temporal topologies against target neuro-states (e.g., wake transitions, spindles). By introducing a classification penalty into the generator's optimization gradient, the model avoids local minima that produce noise, locking the temporal synthesis around biologically valid transition matrices.

### 3.2 Phase 2: Cross-Modal Semantic-TimeGAN

Having guaranteed structural EEG validity, Phase 2 implements a zero-shot retrieval framework that enforces a bijective mapping between the physiological state space and continuous linguistic meaning. We utilize a pre-trained contextual text encoder (e.g., MiniLM-based Transformer) to project free-form affective language reports regarding dream content into a dense hyperspherical embedding space, $\mathcal{Y}_{emb} \in \mathbb{R}^{D}$.

Semantic-TimeGAN freezes the recurrent parameters optimized in Phase 1 and introduces a deep contrastive projection head, $P_{sim}$. This nonlinear transformation operates on the extracted physiological states $\mathbf{h}_i$ to align them with their corresponding linguistic vectors $\mathbf{y}_i$. The semantic grounding is mathematically enforced using the InfoNCE contrastive objective:
$$ \mathcal{L}_{InfoNCE} = - \mathbb{E} \left[ \log \frac{\exp(sim(P_{sim}(\mathbf{h}_i), \mathbf{y}_i) / \tau)}{\sum_{j=1}^{K} \exp(sim(P_{sim}(\mathbf{h}_i), \mathbf{y}_j) / \tau)} \right] $$
where $sim(\cdot, \cdot)$ denotes cosine similarity and $\tau$ represents the temperature scaling hyperparameter. 

By maximizing the inner product similarity between corresponding physiological-semantic pairs and actively pushing apart mismatched topologies, Semantic-TimeGAN generates an organized multidimensional cognitive manifold. The strict sequential execution guarantees that textual conditioning only steers the sampling probabilities across valid physiological distributions, never violating the intrinsic neurobiology established in Phase 1.

## 4. Experimental Setup

### 4.1 Dataset Cohorts

The framework was statistically quantified across two distinct empirical cohorts to separately validate physiological generation and cross-modal semantic mapping.

**1. Physiological Benchmarking:** Robustness and structural validity were evaluated utilizing sleep recordings from physical polysomnography databases, prominently encompassing the PhysioNet Sleep-EDF configuration. These datasets provide prolonged temporal recordings of multi-channel sensor arrays, enabling calculation of prolonged transitional entropies, spectral scaling decay, and localized vertex waves.

**2. Semantic Alignment Verification:** To strictly evaluate semantic conditioning, we deployed the Multimodal Dataset for Dream Emotion Classification (DEED) \cite{zheng2022deed}. The analysis specifically utilized the 654.84 MB subset comprising tightly cropped dream EEG clips mapped deterministically to emotional text labels. Extracting the prolonged 37.85 GB whole-night raw continuous data was mathematically contraindicated; maintaining a strict temporal correlation radius between the cognitive report boundary and the physiological recording window is mandatory to prevent massive semantic dilution in the contrastive projection space.

### 4.2 Evaluation Metrics

The architecture was benchmarked utilizing two strictly decoupled metric taxonomies:

**Physiological Fidelity Metrics:** 
To verify that the output of AC-TimeGAN does not mimic random noise, structural comparisons were mapped. Power Spectral Density (PSD) deviations were calculated across canonical frequency bands (Delta 0.5-4Hz, Theta 4-8Hz, Alpha 8-12Hz, Beta 13-30Hz). Weighted macro-accuracy and F1-score from a discrete downstream classifier evaluated the non-trivial microstate structure present within the synthetic signals.

**Semantic Alignment Metrics:** 
Phase 2 quantification relied heavily on retrieving matching text-brainwave pairs without supervised fine-tuning. We computed the Mean Cosine Similarity across the aligned latent manifold. Final zero-shot cross-modal retrieval capacity was rigorously tabulated using Hits@1 (Top-1 Accuracy) and Hits@3 (Top-3 Accuracy) configurations. Furthermore, t-Stochastic Neighbor Embedding (t-SNE) topologies were generated to qualitatively analyze the spatial cohesion of distinct semantic-affective clusters distributed across the synthesized biosignals.
