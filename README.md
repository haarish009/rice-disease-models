# rice-disease-models
# 2S-XAI-CNN: Stage-Wise Attention-Guided CNN for High-Accuracy Rice Leaf Disease Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange)](https://tensorflow.org)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

[cite_start]An open-source implementation of the **2-Stage Explainable Convolutional Neural Network (2S-XAI-CNN)**[cite: 12]. [cite_start]This framework bridges the gap between deep-learning diagnostic performance and model transparency in precision agriculture[cite: 49]. [cite_start]By utilizing a decoupled "coarse-to-fine" training paradigm [cite: 259] [cite_start]on a lightweight **MobileNetV2** backbone [cite: 77][cite_start], this architecture achieves state-of-the-art accuracy while maintaining a low computational footprint viable for mobile and edge application deployments[cite: 12, 391, 392].

---

## 📖 Table of Contents
1. [Key Features](#-key-features)
2. [Architectural Overview](#%EF%B8%8F-architectural-overview)
3. [Dataset & Data Pipeline](#-dataset--data-pipeline)
4. [Mathematical Foundations](#-mathematical-foundations)
5. [Installation & Setup](#-installation--setup)
6. [Training Pipeline](#-training-pipeline)
7. [Explainability via Grad-CAM](#-explainability-via-grad-cam)
8. [Experimental Results](#-experimental-results)
9. [Citation & References](#-citation--references)

---

## 🌟 Key Features

* [cite_start]**Two-Stage Fine-Tuning Pipeline:** Decouples global feature extraction ($LR = 10^{-3}$) from fine-grained inter-class boundary optimization ($LR = 10^{-4}$) to solve severe visual mimesis anomalies[cite: 71, 79, 299, 335, 336].
* [cite_start]**Edge-Optimized Backbone:** Built upon an inverted residual MobileNetV2 architecture, optimizing accuracy-to-parameter constraints[cite: 77, 296].
* [cite_start]**Custom Memory Allocation Layer:** Integrates a thread-safe `MemoryEfficientGenerator` to facilitate high-volume training without host RAM exhaustion[cite: 280, 281].
* [cite_start]**Explainable AI (XAI) Verifier:** Fully integrated Grad-CAM diagnostic layer mapping localized feature updates directly to network predictions, protecting the network from background correlation artifacts ("Clever Hans" effect)[cite: 80, 313, 377, 378].

---

## 🗺️ Architectural Overview

The network decomposes feature spaces into two distinct optimization loops:

[Input Image] ➔ [Data Augmentation] ➔ [Stage 1: Frozen MobileNetV2 Base] ➔ [Initial Weight Mapping]│[Classification Output] 🔀 [Grad-CAM Masking] 🌁 🤹 [Stage 2: Top-N Unfrozen Layer Update]
1.  **Stage-1 (Structural Feature Extraction):** The MobileNetV2 backbone is frozen[cite: 308]. The network focuses on learning structural plant morphology, isolating leaf matrices, and general lesion boundaries[cite: 13, 300].
2.  **Stage-2 (Boundary Optimization):** Deep blocks are selectively unfrozen[cite: 309, 311]. The model fine-tunes decision borders among visually highly similar diseases (e.g., *Leaf Blast* vs. *Brown Spot*)[cite: 16, 363].

---

## 📊 Dataset & Data Pipeline

This repository is calibrated against the **Paddy Doctor Dataset / Kaggle Rice Leaf Diseases Detection benchmark**, featuring a multi-class configuration across **10 disease conditions** plus a **Healthy class**[cite: 87, 276, 277, 357]:

* Bacterial Leaf Blight [cite: 358]
* Brown Spot [cite: 358]
* Healthy [cite: 358]
* Leaf Blast [cite: 358]
* Leaf Scald [cite: 358]
* Narrow Brown Spot [cite: 358]
* Neck Blast [cite: 358]
* Rice Hispa [cite: 358]
* Sheath Blight [cite: 358]
* Tungro [cite: 358]

### Online Execution Profiling
To mitigate high storage strain, the custom sequence generator enforces:
* Dynamic single-batch disk reads via lazily evaluating native local image lists[cite: 283].
* Spatial constraints downscaled uniformly to $160 \times 160 \times 3$ channels[cite: 282, 294].
* Dynamic matrix scaling ($[0, 1]$ normalization) and stochastic geometric transforms (horizontal/vertical flips)[cite: 286, 288].

---

## 🧮 Mathematical Foundations

### 1. 2D Convolution Core
For input sub-matrix space $I$, the transformation kernel mapping layer outputs $S$ at spatial index $(i, j)$ runs via:

$$S(i,j) = (I * K)(i,j) = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c=0}^{C-1} I_{i+m, j+n, c} \cdot K_{m, n, c} + b$$ [cite: 191, 192]

### 2. Dimension Collapse (Global Average Pooling)
To prevent fully connected flattening from overparameterizing edge node layers, the final feature maps are collapsed spatially using Global Average Pooling (GAP):

$$\text{GAP}_k = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} S_{k}(i, j)$$ [cite: 231, 232]

### 3. Grad-CAM Neuron Localization Matrix
Target class activation weighting maps utilize linear gradient back-propagation maps across pooled spatial areas to construct contextual weight bounds $\alpha_k^c$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$ [cite: 318, 320]

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.9 or higher [cite: 5]
* CUDA Toolkit 11.8+ (for GPU accelerated training loops)

### Dependency Aggregation
Clone the repository and install requirements via pip:
```bash
git clone [https://github.com/username/2S-XAI-CNN.git](https://github.com/username/2S-XAI-CNN.git)
cd 2S-XAI-CNN
pip install -r requirements.txt
Create a requirements.txt containing:Plaintexttensorflow>=2.10.0
numpy>=1.22.0
opencv-python>=4.6.0
matplotlib>=3.5.2
scikit-learn>=1.1.1
🚀 Training PipelineExecute the full two-stage training loop end-to-end:Pythonfrom model.pipeline import TwoStageTrainer

# Initialize configuration using paper constraints
trainer = TwoStageTrainer(
    image_size=(160, 160),     # [cite: 282, 333]
    batch_size=16,             # [cite: 282, 334]
    stage1_epochs=20,          # [cite: 297, 335]
    stage2_epochs=20,          # [cite: 335]
    lr_stage1=1e-3,            # [cite: 299, 335]
    lr_stage2=1e-4             # [cite: 336]
)

# Execute core training phases
trainer.run_stage_one(dataset_path="./data/rice_leaf_diseases")
trainer.run_stage_two()
trainer.evaluate_test_set()
To run using the terminal script interface:Bashpython train.py --data_dir ./data/rice_leaf_diseases --output_dir ./saved_models
🌁 Explainability via Grad-CAMTo produce activation heatmaps for validation checkpoints, compile inference masks through the native visualization framework:Bashpython explain.py --image_path ./data/test/leaf_blast_sample.jpg --model_path ./saved_models/2s_xai_cnn.hsi
This generates a three-panel composite visualization overlay output displaying:Original Sample Matrix Image   Attention Mask Alignment Boundary   Alpha-Weighted Superimposed Overlay Segment Map   📊 Experimental ResultsStage Progression AnalyticsThe transition from Stage-1 to Stage-2 fine-tuning results in a significant reduction in overall error rates:  Model Model Loop ConfigurationValidation AccuracyEvaluation Test Set AccuracyArchitectural Diagnostic ProfilingStage-1 (MobileNetV2 Base)94.50%94.10%Strong baseline; vulnerable to mimic cross-talk.  Stage-2 (2S-XAI-CNN Proposed)98.80%98.60%State-of-the-art fine-grained boundary resolution (+4.5%).  Robust Class Diagnostics Metrics SummaryThe model achieves balanced performance across all pathological targets:  Bacterial Leaf Blight: 1.00 F1-Score   Tungro Virus: 1.00 F1-Score   Leaf Blast: 0.95 F1-Score   Brown Spot: 0.97 F1-Score   📄 Citation & ReferencesIf you build upon this architecture or use this code in your research, please cite the primary work:Code snippet@article{jansi2026stagewise,
  title={Stage-Wise Attention-Guided CNN (2S-XAI-CNN) for High-Accuracy Rice Leaf Disease Classification},
  author={Jansi, R. and Haarish, S.},
  affiliation={Department of Electronics and Communication Engineering, SRM Institute of Science and Technology},
  year={2026}
}
Reference BaselinesPetchiammal et al. (2023) - Paddy Doctor Visual Image Dataset Benchmarking.  Padhi et al. (2024) - EfficientNet B4 Agricultural Compound Scaling.  Xu et al. (2025) - LiSA-MobileNetV2 Lightweight Swish Adaptations.  
---

### Key Components to include in your project folder structure:
To make this repository fully functional, ensure your folder structure follows this layout:
```text
2S-XAI-CNN/
├── data/
│   └── rice_leaf_diseases/      # Place Kaggle class folders here
├── model/
│   ├── __init__.py
│   ├── generator.py             # Contains MemoryEfficientGenerator class
│   ├── network.py               # Contains MobileNetV2 architecture logic
│   └── pipeline.py              # Handles Stage 1 and Stage 2 optimization loops
├── saved_models/
├── train.py                     # Main execution entrypoint script
├── explain.py                   # Contains Grad-CAM extraction logic
└── requirements.txt
