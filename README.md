# ODIR Multimodel LDA BiLSTM

A deep learning pipeline for automated ocular disease classification using the ODIR-5K dataset. Combines multi-model transfer learning feature extraction, Linear Discriminant Analysis (LDA) for dimensionality reduction, and DNN / LSTM / BiLSTM classifiers — achieving **~85% validation accuracy** across 8 disease classes.

> Implements and extends the methodology from:  
> Kansal et al. (2025). *Multiple model visual feature embedding and selection method for an efficient ocular disease classification.* Scientific Reports, 15, 5157. https://doi.org/10.1038/s41598-024-84922-y

---

## Project Overview

Eye diseases like diabetic retinopathy, glaucoma, cataract, and age-related macular degeneration are leading causes of preventable blindness worldwide. Manual diagnosis is time-consuming, specialist-dependent, and inaccessible in resource-limited settings. This project builds an automated classification pipeline using fundus images from the ODIR-5K dataset.

The pipeline runs in two phases:

**Phase 1 — Transfer Learning Baseline**  
Three pretrained CNN models (DenseNet201, EfficientNetB3, InceptionResNetV2) are fine-tuned on ODIR fundus images to establish a baseline and to learn domain-adapted feature representations.

**Phase 2 — LDA Feature Selection + Sequential Classifiers**  
Features extracted from all three CNNs are combined into a 9,696-dimensional vector. LDA (Linear Discriminant Analysis) reduces this to 21 discriminative components (7 per model). DNN, LSTM, and BiLSTM classifiers are then trained on these compact features.

---

## Key Results

| Classifier | Feature set | Val Accuracy | Val Precision | Val Recall |
|---|---|---|---|---|
| DNN | Combined (21-dim) | ~75% | ~82% | ~71% |
| LSTM | Combined (21-dim) | ~78% | ~80% | ~72% |
| **BiLSTM** | **Combined (21-dim)** | **~85%** | **~85%** | **~84%** |
| DenseNet201 (TL baseline) | Raw images | ~55% | ~64% | ~43% |
| EfficientNetB3 (TL baseline) | Raw images | ~62% | ~66% | ~55% |
| InceptionResNetV2 (TL baseline) | Raw images | ~58% | ~65% | ~52% |

BiLSTM with combined LDA features consistently outperforms all individual transfer learning baselines and single-model feature strategies.

---

## Features

- Stratified 80/20 train/validation split with class-aware oversampling for rare disease classes
- Multi-model feature extraction (DenseNet201 · EfficientNetB3 · InceptionResNetV2) using `timm`
- Partial layer unfreezing for domain fine-tuning
- LDA-based dimensionality reduction: 9,696 raw features → 21 discriminative components
- Three classifier architectures compared: DNN, LSTM, BiLSTM (all in PyTorch)
- Weighted cross-entropy loss to handle class imbalance
- Early stopping and ReduceLROnPlateau learning rate scheduling
- Grad-CAM heatmap visualizations for model explainability
- Full training curves, confusion matrix, and per-class metrics

---

## Project Structure

```
odir-multimodel-lda-bilstm/
├── README.md                    # Project documentation
├── project.ipynb   # Full implementation notebook
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
```

---

## Dataset

**ODIR-5K — Ocular Disease Intelligent Recognition**  
Source: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

The dataset contains colour fundus photographs of both eyes for 5,000 patients, labelled across 8 categories:

| Label | Disease | Approx. samples |
|---|---|---|
| N | Normal | 2,873 |
| D | Diabetes | 1,608 |
| O | Other diseases | 708 |
| M | Pathological Myopia | 708 |
| G | Glaucoma | 284 |
| C | Cataract | 293 |
| A | Age-related Macular Degeneration | 266 |
| H | Hypertension | 128 |

Download the dataset from Kaggle and place it so the notebook can locate the `full_df.csv` annotations file and the fundus image folder. The notebook auto-detects the folder structure.

---

## Model Architecture

### Phase 1 — Transfer Learning Feature Extractors

All three models are loaded with ImageNet weights via `timm` and partially fine-tuned:

| Model | Input size | Extracted features | Fine-tuned layers |
|---|---|---|---|
| DenseNet201 | 299×299 | 1,920 | denseblock4 + norm5 |
| EfficientNetB3 | 299×299 | 1,536 | features[6:] |
| InceptionResNetV2 | 299×299 | 6,240 (3 × Block8 hooks) | Last 3 Block8 layers |

Combined raw feature vector: **9,696 dimensions**

### Phase 2 — LDA + Sequential Classifiers

LDA (sklearn `LinearDiscriminantAnalysis`, n_components=7) is applied per-model, and the 7-component outputs are concatenated: **3 × 7 = 21 features** fed to classifiers.

**DNN:**
```
Input(21) → FC(128)+BN+ReLU+Drop(0.4) → FC(64)+BN+ReLU+Drop(0.3)
         → FC(32)+ReLU+Drop(0.2) → FC(8) → Softmax
```

**LSTM:**
```
Input(21) → unsqueeze(1) → LSTM(128) → LSTM(64)
          → last timestep → BN+Drop(0.4) → FC(64→32)+ReLU → FC(32→8)
```

**BiLSTM:**
```
Input(21) → unsqueeze(1) → BiLSTM(128, bidir→256) → BiLSTM(64, bidir→128)
          → last timestep → BN(128)+Drop(0.4) → FC(128→32)+ReLU → FC(32→8)
```

---

## Installation

### Requirements

- Python 3.7+
- CUDA-capable GPU (recommended; CPU fallback supported)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/iam-mukeshvenkatesh/odir-multimodel-lda-bilstm.git
cd odir-multimodel-lda-bilstm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the ODIR-5K dataset from Kaggle and place it in the project root or update the dataset path in the notebook.

5. Launch the notebook:
```bash
jupyter notebook latest-project-code.ipynb
```

### Core Dependencies

```
torch>=2.0
torchvision>=0.15
timm>=0.9
scikit-learn>=1.3
numpy
pandas
Pillow
opencv-python
matplotlib
seaborn
jupyter
tqdm
```

---

## Usage

Run the notebook cells sequentially. The pipeline is divided into clearly labelled sections:

1. **Dataset preparation** — loads annotations, builds image-level DataFrame, applies oversampling
2. **Visualization** — class distribution, augmentation previews
3. **Phase 1 training** — fine-tunes all three transfer learning models for 5 epochs each
4. **Feature extraction** — extracts and saves raw feature vectors for train and val sets
5. **LDA feature selection** — fits LDA per model, builds combined 21-dim feature set
6. **Phase 2 training** — trains DNN, LSTM, and BiLSTM with early stopping
7. **Evaluation** — accuracy/F1 heatmap, learning curves, confusion matrix
8. **Explainability** — Grad-CAM heatmaps on sample predictions

---

## Differences from the Base Paper

This implementation diverges from Kansal et al. (2025) in several intentional ways:

| Aspect | Base paper | This implementation |
|---|---|---|
| Framework | TensorFlow / Keras | **PyTorch + timm** |
| InceptionResNetV2 features | 4,608 (GAP layer) | **6,240 (3 × Block8 hooks)** |
| LDA combined strategy | 8,064 raw → LDA → 7 | **Per-model LDA (7 each) → concat → 21** |
| Layer freezing | All frozen | **Partial unfreezing for fine-tuning** |
| Oversampling | Not described | **Manual class multipliers** |
| Classifier regularization | Minimal | **BatchNorm + Dropout + L2 weight decay** |

---

## References

- Kansal, I. et al. (2025). Multiple model visual feature embedding and selection method for an efficient ocular disease classification. *Scientific Reports*, 15, 5157.
- He, K. et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Huang, G. et al. (2017). Densely connected convolutional networks. *CVPR*.
- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking model scaling for CNNs. *ICML*.
- Graves, A. & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM. *IJCNN*.
- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179–188.
- ODIR Dataset: https://odir2019.grand-challenge.org/

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Mukesh Venkatesh**  
GitHub: [@iam-mukeshvenkatesh](https://github.com/iam-mukeshvenkatesh)

---

*Last updated: June 2026*
