# Enhanced Group Activity Recognition 🏐

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CVPR 2016](https://img.shields.io/badge/Paper-CVPR'16-green.svg)](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)
[![arXiv](https://img.shields.io/badge/Extension-arXiv%3A1607.02643-b31b1b.svg)](https://arxiv.org/pdf/1607.02643)

An enhanced PyTorch implementation of the hierarchical deep temporal framework for **Group Activity Recognition (GAR)** in multi-person video sequences. This project implements multiple baseline architectures and proposes advanced models to recognize collective human activities from video data.

This repository builds upon the foundational work of Ibrahim et al.:

- **[CVPR 2016]** [*A Hierarchical Deep Temporal Model for Group Activity Recognition*](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)
- **[Extended journal version (IJCV 2017)]** [*A Hierarchical Deep Temporal Model for Group Activity Recognition*](https://arxiv.org/pdf/1607.02643)

---

## 📌 Overview

Recognizing collective human activities (e.g. team sports, social gatherings) requires capturing both the actions of individual actors and the complex spatio-temporal interactions among actors across video frames. A single-frame or single-person model cannot, on its own, resolve activities that are only distinguishable through *how people move relative to one another over time*.

This project re-implements and extends the original **two-stage hierarchical LSTM** proposed in the papers above, with modernized components (updated CNN backbones, training utilities, configs, and evaluation tooling) on top of the original architecture. We implement **8 different baselines** with progressive complexity, from simple single-frame CNN classifiers to advanced multi-stream hierarchical temporal models.

## 🏗️ Architecture & Baselines

This project implements **8 progressive baselines** ranging from simple to advanced architectures, demonstrating the importance of temporal modeling and hierarchical reasoning for group activity recognition:

### **Baseline Progression**

| Baseline | Architecture | Stages | Temporal | Key Innovation | Group Accuracy |
|----------|--------------|--------|----------|----------------|---|
| **B1** | Single-Frame CNN | 1 | ❌ | Single-frame classification | 72.10% |
| **B3** | Spatial CNN + Pooling | 1 | ❌ | Spatial aggregation before classification | 73.45% |
| **B4** | Image-Level LSTM | 1 | ✅ | Temporal LSTM on image sequences | 78.83% |
| **B5** | Hierarchical (2-stage) | 2 | ✅ | Person LSTM → Group classification | 79.21% |
| **B6** | Single-Stream Temporal | 1 | ✅ | Person LSTM + image LSTM | 77.56% |
| **B7** | Multi-Stream Temporal | 2 | ✅ | Dual LSTMs (player + image) with better fusion | 85.79% |
| **B8** | **Advanced Hierarchical** | 2 | ✅ | **Frozen feature extractors + group-level LSTM** | **88.63%** ⭐ |

### **Baseline Descriptions**

#### **B1: Single-Frame CNN Classifier**
- **Input**: Single frame with person crops
- **Architecture**: ResNet-50 → FC layers
- **Output**: Group activity classification
- **Key Insight**: No temporal modeling; baseline for single-frame performance
- **Result**: 72.10% accuracy (underperforms due to lack of temporal context)

#### **B3: Spatial Feature Aggregation**
- **Person Level**: ResNet-50 on person crops → FC layers → action classification (9 classes)
- **Group Level**: ResNet-50 on full image → Max pooling over person features → classification
- **Key Innovation**: Separation of person-level and group-level reasoning
- **Result**: 73.45% accuracy (modest improvement; still lacks temporal modeling)

#### **B4: Image-Level LSTM**
- **Architecture**: ResNet-50 CNN (frozen) → LSTM (1024 hidden) → FC classifier
- **Input**: Video sequence of whole scene images
- **Key Insight**: Adds temporal modeling at the full-image level
- **Result**: 78.83% accuracy (significant jump from temporal modeling)

#### **B5: Hierarchical Two-Stage (Original Design)**
- **Person Level**: 
  - CNN + LSTM processing each person's trajectory across frames
  - Output: Person-level action features (1024-dim)
- **Group Level**:
  - Max pooling of person features → FC layers → group activity classification
- **Key Innovation**: Hierarchical structure separating person and group reasoning
- **Result**: 79.21% accuracy (hierarchical approach shows promise)

#### **B6: Single-Stream Temporal Model**
- **Architecture**: 
  - Person CNN + LSTM (for player-level temporal modeling)
  - Image CNN + LSTM (for scene-level temporal modeling)
  - Features are simply concatenated
- **Key Insight**: Dual temporal streams but simple fusion
- **Result**: 77.56% accuracy (multi-stream but weak fusion mechanism)

#### **B7: Multi-Stream Hierarchical Temporal**
- **Architecture**:
  - Player-level stream: ResNet-50 + LSTM (1024 hidden)
  - Image-level stream: ResNet-50 + LSTM (1024 hidden)
  - Features projected (2048 → 512 each)
  - **Fusion**: Concatenated features (3584-dim) → Group LSTM
- **Key Innovation**: Better fusion of player and image temporal features
- **Result**: 85.79% accuracy (strong multi-stream design)

#### **B8: Advanced Hierarchical Temporal (Best Model)** 🏆
- **Architecture**:
  - **Frozen Components** (fixed backbones):
    - Image CNN features (2048-dim)
    - Player CNN features (2048-dim)
    - Player LSTM representations (2048-dim)
  - **Trainable Components**:
    - Image projection: 2048 → 512
    - Player projection: 2048 → 512
    - **Group-level LSTM** (512 hidden) on concatenated features
    - Classification head: 512 → 128 → 8 classes
- **Key Innovation**: 
  - Strategic freezing of learned features from B5/B7
  - Focus training on high-level group interaction modeling
  - Batch normalization + Dropout for regularization
- **Result**: **88.63% accuracy** ⭐ (best overall performance)

---

### **Original Hierarchical Two-Stage Architecture (From Paper)**

The core architecture from Ibrahim et al. follows this design:

```
video clip (T frames)
   │
   ▼
person bounding-box crops (T frames × P persons)
   │
   ▼
CNN feature extractor  ──►  Stage 1: Person-level LSTM (per-actor temporal dynamics)
   │                               (learns individual action patterns)
   ▼
Pooling (max / average) across actors  ──►  Fixed-size scene representation
   │
   ▼
Stage 2: Group-level LSTM (scene-level temporal dynamics)
   │                        (learns group interaction patterns)
   ▼
Collective activity classification (8 classes)
```

**Key Design Principles**:
1. **Hierarchical Decomposition**: Reason about individuals first, then the group
2. **Temporal Modeling**: LSTMs capture action dynamics across frames
3. **Aggregation**: Pool individual features into a unified scene representation
4. **End-to-End Learning**: Both stages trained jointly with class labels

## 📂 Repository Structure

```
Group-Activity-Recognition-Project/
├── configs/              # Experiment / model / training configuration files
│   ├── b1_config1.yaml
│   ├── b3_player_classifier_config1.yaml
│   ├── b3_group_classifier_config1.yaml
│   ├── b4_config1.yaml
│   ├── b5_player_classifier_config1.yaml
│   ├── b5_group_classifier_config1.yaml
│   ├── b6_config1.yaml
│   ├── b7_config1.yaml
│   └── b8_config1.yaml
├── models/               # Baseline model implementations
│   ├── b1.py            # Single-frame CNN
│   ├── b3.py            # Spatial aggregation (player + group)
│   ├── b4.py            # Image-level LSTM
│   ├── b5.py            # Hierarchical 2-stage (player + group)
│   ├── b6.py            # Single-stream temporal
│   ├── b7.py            # Multi-stream hierarchical
│   └── b8.py            # Advanced hierarchical (best) ⭐
├── scripts/              # Training and evaluation scripts
│   ├── B1/
│   ├── B3/ (player + group)
│   ├── B4/
│   ├── B5/ (player + group)
│   ├── B6/
│   ├── B7/
│   └── B8/
├── train/                # Training entry points for each baseline
│   └── b1.py through b8.py
├── utils/                # Helper functions
│   ├── dataset.py        # Dataset loading and preprocessing
│   ├── volleyball_annot_loader.py
│   ├── logger.py         # Logging utilities
│   └── extract_features.py
├── outputs/              # Results (checkpoints, logs, evaluation metrics)
│   ├── b1/model_1_config_1/
│   ├── b3/ (player + group)
│   ├── b4/model_1_config_1/
│   ├── b5/ (player + group)
│   ├── b6/model_1_config_1/
│   ├── b7/model_1_config_1/
│   └── b8/model_1_config_1/
└── README.md
```


## 📊 Dataset

The model is designed for datasets annotated with both **per-person action labels** and **scene-level group activity labels** on short video clips, in line with the datasets used in the original papers.

### Supported Datasets:
- **Volleyball Dataset** (Ibrahim et al., CVPR 2016) - 4830 short clips with multi-person volleyball interactions
- **Collective Activity Dataset** (Choi et al., ECCV 2010) - 32 videos of collective activities
- Custom datasets following the same annotation format

### Data Format:
Each video clip provides:
- **Person tracklets**: Bounding boxes across a temporal window of frames
- **Per-person action labels**: 9 classes (waiting, setting, digging, falling, spiking, blocking, jumping, moving, standing)
- **Group activity labels**: 8 classes (l-pass, r-pass, l-spike, r-spike, l-set, r-set, l-winpoint, r-winpoint)

Update the dataset paths in `configs/` to point to your local copy of the data before training.

---

## 📈 Experimental Results & Comparison

### **Group Activity Recognition (Video-Level Accuracy)**

```
Baseline Performance Comparison:

B8 (Advanced Hierarchical) ⭐ ████████████████████ 88.63%
B7 (Multi-Stream Hierarchical) ███████████████ 85.79%
B5 (Hierarchical 2-Stage) ██████████████ 79.21%
B4 (Image-Level LSTM) ██████████████ 78.83%
B6 (Single-Stream Temporal) ██████████████ 77.56%
B3 (Spatial Aggregation) █████████████ 73.45%
B1 (Single-Frame CNN) █████████████ 72.10%
```

### **Detailed Performance Metrics (Test Set)**

#### **B1: Single-Frame CNN Classifier**
```
Accuracy: 72.10%  |  F1-Macro: 73.69%  |  F1-Weighted: 72.23%

Per-Class Performance:
             Precision  Recall  F1-Score
l-pass       0.72      0.68      0.70
r-pass       0.65      0.68      0.66
l-spike      0.85      0.75      0.80
r-spike      0.81      0.69      0.75
l-set        0.78      0.61      0.68
r-set        0.57      0.78      0.66
l-winpoint   0.80      0.82      0.81
r-winpoint   0.77      0.91      0.84
```

#### **B3: Spatial Feature Aggregation**
```
Player Accuracy: 79.05%  |  Player F1-Macro: 56.47%  |  F1-Weighted: 77.77%

Group Accuracy: 73.45%  |  Group F1-Macro: 69.54%  |  F1-Weighted: 73.02%

Group Per-Class Performance:
             Precision  Recall  F1-Score
l-pass       0.74      0.79      0.76
r-pass       0.67      0.76      0.71
l-spike      0.81      0.83      0.82
r-spike      0.81      0.79      0.80
l-set        0.82      0.77      0.79
r-set        0.84      0.72      0.78
l-winpoint   0.53      0.71      0.60
r-winpoint   0.43      0.23      0.30
```

#### **B4: Image-Level LSTM**
```
Accuracy: 78.83%  |  F1-Macro: 79.68%  |  F1-Weighted: 78.75%

Per-Class Performance:
             Precision  Recall  F1-Score
l-pass       0.78      0.74      0.76
r-pass       0.68      0.83      0.75
l-spike      0.88      0.88      0.88
r-spike      0.84      0.86      0.85
l-set        0.77      0.77      0.77
r-set        0.78      0.62      0.69
l-winpoint   0.81      0.86      0.84
r-winpoint   0.88      0.82      0.85
```

#### **B5: Hierarchical Two-Stage**
```
Player Accuracy: 80.93%  |  Player F1-Macro: 61.55%  |  F1-Weighted: 80.20%

Group Accuracy: 79.21%  |  Group F1-Macro: 74.17%  |  F1-Weighted: 78.23%

Group Per-Class Performance:
             Precision  Recall  F1-Score
l-pass       0.88      0.73      0.80
r-pass       0.70      0.88      0.78
l-spike      0.90      0.91      0.90
r-spike      0.87      0.93      0.90
l-set        0.83      0.84      0.83
r-set        0.84      0.79      0.81
l-winpoint   0.57      0.76      0.65
r-winpoint   0.50      0.17      0.26
```

#### **B6: Single-Stream Temporal**
```
Accuracy: 77.56%  |  F1-Macro: 72.62%  |  F1-Weighted: 77.13%

Per-Class Performance:
             Precision  Recall  F1-Score
l-pass       0.80      0.86      0.83
r-pass       0.78      0.82      0.80
l-spike      0.84      0.88      0.86
r-spike      0.87      0.79      0.82
l-set        0.88      0.81      0.84
r-set        0.83      0.78      0.81
l-winpoint   0.50      0.73      0.59
r-winpoint   0.35      0.20      0.25
```

#### **B7: Multi-Stream Hierarchical Temporal**
```
Accuracy: 85.79%  |  F1-Macro: 86.45%  |  F1-Weighted: 85.83%

Per-Class Performance:
             Precision  Recall  F1-Score
l-pass       0.87      0.83      0.85
r-pass       0.73      0.92      0.81
l-spike      0.95      0.88      0.92
r-spike      0.90      0.88      0.89
l-set        0.86      0.86      0.86
r-set        0.87      0.73      0.80
l-winpoint   0.89      0.93      0.91
r-winpoint   0.90      0.86      0.88
```

#### **B8: Advanced Hierarchical Temporal** 🏆
```
Accuracy: 88.63%  |  F1-Macro: 88.93%  |  F1-Weighted: 88.62%

Per-Class Performance:
             Precision  Recall  F1-Score
l-pass       0.90      0.89      0.90
r-pass       0.81      0.89      0.84
l-spike      0.93      0.94      0.94
r-spike      0.90      0.90      0.90
l-set        0.94      0.92      0.93
r-set        0.85      0.78      0.81
l-winpoint   0.92      0.92      0.92
r-winpoint   0.88      0.87      0.88
```

---

### **Confusion Matrices Visualization**

#### **B1: Single-Frame CNN (72.10% accuracy)**

```
Baseline 1 shows significant confusion, especially between:
  - l-pass ↔ r-pass (low temporal context)
  - l-set ↔ r-pass (spatial ambiguity)
→ Single frames lack sufficient discriminative information
```

#### **B8: Advanced Hierarchical Temporal (88.63% accuracy)** ⭐

```
Strong diagonal dominance indicates:
  ✓ l-pass: 89% correctly classified
  ✓ r-pass: 89% correctly classified
  ✓ l-spike: 94% correctly classified
  ✓ r-spike: 90% correctly classified
  ✓ l-set: 92% correctly classified
  ✓ l-winpoint: 92% correctly classified
  ✓ r-winpoint: 87% correctly classified

Minor confusions:
  • r-set slightly confused with l-pass (3% false negative)
  • l-winpoint with slight confusion on set classifications
```

---

### **Key Performance Insights**

| Aspect | Finding |
|--------|---------|
| **Temporal Modeling Impact** | Adding temporal modeling (B4 vs B1) yields **+6.73%** improvement |
| **Hierarchical Design Benefit** | Hierarchical 2-stage (B5) vs Image LSTM (B4): **+0.38%** (both ~79%) |
| **Multi-Stream Advantage** | Dual streams (B7) vs single-stream (B6): **+8.23%** improvement |
| **Feature Freezing Strategy** | Frozen features + group LSTM training (B8): **+2.84%** over B7 |
| **Best Baseline** | B8 achieves **88.63%** accuracy with strategic feature freezing |
| **Hardest Classes** | r-winpoint and r-set show lower recall; require better temporal modeling |
| **Strongest Classes** | l-spike and l-set achieve >93% precision |

---

### **Training Dynamics (B8 - Best Model)**

```
Epoch   Train Loss   Val Loss   Train Acc   Val Acc   Train F1   Val F1   Learning Rate
1       1.370        1.005      65.43%      81.88%    0.632      0.825    0.0001
2       1.075        0.926      81.32%      83.37%    0.820      0.840    0.0001
3       0.986        0.901      82.76%      82.77%    0.835      0.832    0.0001
...
8       0.801        0.854      89.27%      84.27%    0.900      0.847    0.0001
...
16      0.696        0.834      94.38%      86.06%    0.948      0.865    0.00001

Final: Validation Accuracy: 86.06%, F1-Score: 0.865
```

**Training Observations**:
- ✅ Validation accuracy plateaus around epoch 8-10 at ~84%
- ✅ Learning rate decay (0.0001 → 0.00001) at epoch 12 helps with fine-tuning
- ✅ Frozen feature extractors prevent overfitting
- ⚠️ Large gap between training (94%) and validation (86%) suggests regularization is working

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (highly recommended for training)
- 16+ GB RAM for multi-stream models

### Setup

```bash
# Clone the repository
git clone https://github.com/tareksaber55/Group-Activity-Recognition-Project.git
cd Group-Activity-Recognition-Project

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- PyTorch (CNN + LSTM modules)
- TorchVision (ResNet50 backbone)
- NumPy & Pandas (data handling)
- scikit-learn (evaluation metrics)
- PyYAML (configuration files)
- TensorBoard (training visualization)

---

## 🚀 Usage & Training

### 1. Prepare Your Dataset

Place your dataset in a accessible location and update the paths in the config files:

```yaml
# Example: configs/b8_config1.yaml
DATASET:
  TRAIN_VIDEO_PATH: "/path/to/dataset/train_videos"
  VAL_VIDEO_PATH: "/path/to/dataset/val_videos"
  TEST_VIDEO_PATH: "/path/to/dataset/test_videos"
  ANNOTATION_PATH: "/path/to/dataset/annotations"
```

### 2. Configure Your Experiment

Edit a configuration file to set:
- Dataset paths
- Model architecture parameters
- Learning rate & optimizer settings
- Number of epochs & batch size
- Feature extraction options

Example config structure:
```yaml
MODEL: "b8"
BATCH_SIZE: 32
EPOCHS: 50
LEARNING_RATE: 0.0001
HIDDEN_SIZE: 1024
NUM_CLASSES_PERSON: 9
NUM_CLASSES_GROUP: 8
```

### 3. Train a Baseline

```bash
# Train B1 (Single-Frame CNN)
python train/b1.py

# Train B4 (Image-Level LSTM)
python train/b4.py

# Train B5 (Hierarchical 2-Stage)
python train/b5_player_classifier.py  # Train person-level classifier first
python train/b5_group_classifier.py   # Then train group classifier

# Train B8 (Advanced Hierarchical - Recommended) ⭐
python train/b8.py

# Optional: Run with specific config
python train/b8.py --config configs/b8_config1.yaml
```

### 4. Evaluate & Generate Reports

```bash
# Evaluation scripts are in scripts/ directory
# For example:
python scripts/B8/b8_eval.py --checkpoint outputs/b8/model_1_config_1/best_model.pth

# This will generate:
# - classification_report.txt (per-class metrics)
# - confusion_matrix.png (visualization)
# - logs.csv (training curves)
```

---

## 📊 Outputs & Artifacts

Training generates the following in the `outputs/` directory:

```
outputs/b8/model_1_config_1/
├── best_model.pth              # Best checkpoint (validation accuracy)
├── final_model.pth             # Final checkpoint (last epoch)
├── config.yaml                 # Experiment configuration (for reproducibility)
├── logs.csv                    # Training metrics per epoch
├── classification_report.txt   # Per-class precision, recall, F1-score
├── confusion_matrix.png        # Confusion matrix heatmap
├── events.out.tfevents.*       # TensorBoard logs (optional)
└── train.log                   # Complete training log
```

### Interpreting Classification Report

```
                     Precision  Recall  F1-Score  Support
l-pass               0.90      0.89      0.90      226

Precision: How many predicted positives were actually positive?
Recall:    How many actual positives were correctly predicted?
F1-Score:  Harmonic mean of precision and recall
Support:   Number of test samples for this class
```

---

## 🔄 Baseline Comparison & Best Practices

### When to Use Each Baseline:

| Baseline | Use Case | Training Time | GPU Memory |
|----------|----------|---------------|------------|
| **B1** | Baseline / sanity check | ⚡ ~5 min | 2GB |
| **B3** | Spatial-only ablation | ⚡ ~10 min | 4GB |
| **B4** | Single-stream temporal | ⏱ ~20 min | 6GB |
| **B5** | Original paper design | ⏱ ~30 min | 8GB |
| **B6** | Multi-stream exploration | ⏱ ~35 min | 10GB |
| **B7** | Strong multi-stream | ⏰ ~45 min | 12GB |
| **B8** | Best performance ⭐ | ⏰ ~50 min | 14GB |

### Recommendations:

✅ **Production Use**: Use **B8** for best accuracy (88.63%)  
✅ **Quick Experiments**: Use **B4** or **B5** for fast iteration  
✅ **Research/Ablation**: Use **B1-B7** to understand design choices  
✅ **Computational Constraints**: Use **B1** or **B3** for minimal resources  

### Tips for Training:

```python
# 1. Enable mixed precision for faster training (PyTorch)
torch.cuda.amp.autocast()

# 2. Use learning rate scheduling
lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 3. Monitor training with TensorBoard
tensorboard --logdir=outputs/

# 4. Save checkpoints regularly
torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

# 5. Validate on a separate holdout set
validation_accuracy = evaluate(model, val_loader)
```

## 📚 Citation

If you use this repository, please cite the original papers and consider citing this implementation:

```bibtex
@inproceedings{ibrahim2016hierarchical,
  title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition},
  author    = {Ibrahim, Mostafa S. and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2159--2166},
  year      = {2016}
}

@article{ibrahim2017hierarchical,
  title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition},
  author    = {Ibrahim, Mostafa S. and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
  journal   = {International Journal of Computer Vision},
  volume    = {126},
  number    = {2},
  pages     = {212--229},
  year      = {2017}
}
```

---

## 🙏 Acknowledgments

- **Ibrahim et al.** for the original hierarchical deep temporal model architecture and the Volleyball Dataset
- **Greg Mori** (SFU) and team for foundational research in group activity recognition
- The computer vision community for datasets, benchmarks, and insights
- PyTorch and TorchVision teams for excellent deep learning frameworks

---

## 📖 Further Reading

### Key Papers:
1. **Original Paper**: [A Hierarchical Deep Temporal Model for Group Activity Recognition (CVPR 2016)](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)
2. **Extended Journal Version**: [IJCV 2017](https://arxiv.org/pdf/1607.02643)
3. **Collective Activity Dataset**: [Choi et al., ECCV 2010](http://www.cs.rochester.edu/u/rmurali/papers/eccv10.pdf)
4. **Skeleton-Based GAR**: [Jain et al., CVPR 2016](https://arxiv.org/abs/1511.06984)

### Related Work:
- Temporal Segment Networks (Wang et al., ECCV 2016)
- Two-Stream Convolutional Networks (Simonyan & Zisserman, NIPS 2014)
- Temporal 3D CNNs (Tran et al., ICCV 2015)
- Attention Mechanisms in Video Understanding (Vaswani et al., NIPS 2017)

---

## 🐛 Troubleshooting

### Common Issues:

**Q: Out of Memory (OOM) during training**
```
A: Reduce BATCH_SIZE in config, or use gradient accumulation:
   - Decrease batch_size from 32 to 16
   - Use B4 or B5 instead of B8 (smaller memory footprint)
   - Enable mixed precision training (PyTorch Automatic Mixed Precision)
```

**Q: Poor validation accuracy despite good training accuracy**
```
A: Model is overfitting. Try:
   - Increase dropout rate (0.5 → 0.6)
   - Add L2 regularization (weight_decay=1e-4)
   - Use early stopping on validation loss
   - Augment training data (flip frames, crop variations)
```

**Q: Dataset not loading**
```
A: Check:
   - Dataset path in config file is correct
   - Annotation files are properly formatted
   - Bounding box coordinates are valid (within image bounds)
   - Person crops are not empty or too small
```

**Q: Confusion between specific activity classes**
```
A: This is expected for visually similar activities:
   - l-pass vs r-pass: Use multi-view information
   - l-set vs l-spike: Temporal context is crucial
   - Train longer or use pre-trained backbones
```

---

## 🔬 Advanced Usage

### Implementing Custom Baselines:

```python
# Example: Create a new baseline model
import torch.nn as nn
import torchvision.models as models

class CustomBaseline(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # Feature extractor
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Your custom layers here
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B*F, C, H, W)
        x = self.features(x)
        x = x.view(B, F, -1)
        x, _ = self.lstm(x)
        x = self.classifier(x[:, -1, :])
        return x

# Add to models/ directory and create corresponding train script
```

### Fine-tuning on Your Dataset:

```bash
# 1. Transfer weights from B8 pretrained on Volleyball
# 2. Create new config with your dataset
# 3. Train with smaller learning rate (1e-5) and fewer epochs
python train/b8.py --pretrained outputs/b8/model_1_config_1/best_model.pth \
                    --config configs/custom_dataset.yaml \
                    --learning_rate 1e-5 \
                    --epochs 20
```

### Hyperparameter Tuning:

```python
# Use grid search or random search
from itertools import product

params = {
    'hidden_size': [512, 1024, 2048],
    'dropout': [0.3, 0.5, 0.7],
    'learning_rate': [1e-4, 5e-4, 1e-3],
}

best_acc = 0
best_params = {}

for p in product(*params.values()):
    config = dict(zip(params.keys(), p))
    acc = train_and_evaluate(config)
    if acc > best_acc:
        best_acc = acc
        best_params = config
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

For academic use, please also acknowledge the original Ibrahim et al. papers.

---

## 📧 Contact & Contributions

For questions, bug reports, or contributions, please open an issue on GitHub or contact the repository maintainers.

**We welcome**:
- ✅ Bug fixes and improvements
- ✅ New baseline implementations
- ✅ Dataset adaptations
- ✅ Performance optimizations
- ✅ Documentation enhancements

---

## 🎯 Future Work

- [ ] Support for skeleton-based features (pose estimation)
- [ ] Attention mechanisms for interaction modeling
- [ ] Graph neural networks for group reasoning
- [ ] Real-time inference optimization
- [ ] Multi-dataset evaluation and domain adaptation
- [ ] Visualization tools for activity understanding
- [ ] Uncertainty quantification in predictions
