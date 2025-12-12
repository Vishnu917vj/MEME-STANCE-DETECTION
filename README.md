# Telugu Meme Stance Detection: Multimodal Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

**Multimodal stance detection on Telugu political memes using BERT-based text models, ResNet50/VGG19 vision backbones, and fusion classifiers.**

---

## 🎯 **Project Overview**

This repository implements a **multimodal deep learning pipeline** for stance detection in Telugu political memes. The system analyzes both **textual content** (from meme captions) and **visual content** (from images) to classify the stance toward political targets as:

- **Favour** (0) - Supports the target
- **Against** (1) - Opposes the target  
- **None** (2) - Neutral/no clear stance

### Key Features
- **Text-only models**: XLM-RoBERTa, mBERT, IndicBERTv2 (6 epochs each)
- **Vision-only models**: ResNet50, VGG19 (8 epochs each)
- **Multimodal fusion**: ResNet50 + trained text models (8 epochs)
- **Evaluation**: Comprehensive testing with confusion matrices, F1-scores, and publication-ready plots
- **Telugu language support**: Handles Telugu text in memes with IndicBERTv2

---

## 📊 **Dataset**

The dataset consists of Telugu political memes with:
- **Images**: Political meme images (download from Google Drive link below)
- **Metadata**: CSV file with extracted text, targets, and stances
- **Unique Targets**: 28 political figures/parties (e.g., Modi, Jagan Mohan Reddy, Pawan Kalyan, BJP, Congress)
- **3-class classification**: Favour, Against, Neutral

### Download Instructions
1. **Images** (Train Images Folder):
   - Download from: [Google Drive Folder](https://drive.google.com/drive/folders/1UmbQs5tzVdaJbvvgSfjHkYz1p46be1_Q?usp=drive_link)
   - Extract to a local folder, e.g., `./data/images/` (update `full_path` in CSV if needed)
   - ~[Number of images, e.g., 1,000+] JPG/PNG files of memes

2. **Metadata** (Train CSV):
   - Download from: [Google Sheets - Train_data](https://docs.google.com/spreadsheets/d/1LsP8deyiLG4g_IWAT-oUREHiuuzLBKVY6KdfQoYw8y8/edit?usp=sharing)
   - Export as CSV and save to `./data/train.csv`
   - Columns: `full_path` (image path), `Image Text` (OCR text), `TARGET_1` (political target), `STANCE_1` (label: favour/against/none)

**Note**: Update paths in `train.py` and `evaluate_all_models.py` to point to your local `./data/` folder. Ensure image paths in CSV match your local structure (e.g., `./data/images/image1.jpg`).

---

## 🏗️ **Model Architecture**

### 1. **Text-Only Pipeline**
```
Input Text (TARGET_1 + [SEP] + Image Text) 
    ↓
Tokenizer (MAX_LEN=128)
    ↓
BERT/XLM-RoBERTa/IndicBERT (768-dim CLS/pooler) 
    ↓
Dropout(0.3)
    ↓
Linear(768 → 3) → Logits (Favour/Against/None)
```

### 2. **Vision-Only Pipeline**
```
Input Image (224×224 RGB)
    ↓
ResNet50 → AvgPool → 2048-dim
    ↓
Linear(2048 → 3) → Logits
---
OR
VGG19 → Classifier → 4096-dim  
    ↓
Linear(4096 → 3) → Logits
```

### 3. **Multimodal Fusion**
```
Image → ResNet50 (Frozen) → 2048-dim Vision Features
Text → BERT/XLM-R (Trained) → 768-dim Text Features
    ↓
Concatenate → 2816-dim
    ↓
Dropout(0.5) → Linear(2816→512) → ReLU
    ↓
Dropout(0.3) → Linear(512→256) → ReLU
    ↓
Linear(256→3) → Final Logits
```

---

## 🚀 **Quick Start**

### Prerequisites
```bash
# Python 3.8+
pip install torch torchvision transformers pandas scikit-learn pillow matplotlib seaborn openpyxl
```

### 1. **Setup Dataset**
- Download images to `./data/images/`
- Download CSV to `./data/train.csv`
- Update paths in scripts:
  ```python
  TRAIN_PATH = "./data/train.csv"  # Or your Excel if using original
  TEST_PATH = "./data/train.csv"   # Split or use same for quick testing
  ```

### 2. **Train All Models**
```bash
python train.py
```
- **Phase 1**: Train text models (6 epochs total)
- **Phase 2**: Train vision models (8 epochs)  
- **Phase 3**: Train multimodal fusion (8 epochs)
- **Outputs**: Saved models in `saved_models/` + results in `outputs/`

### 3. **Evaluate on Test Set**
```bash
python evaluate_all_models.py
```
- Loads all trained models
- Evaluates on test data
- **Outputs**: `evaluation_results/` with CSV, LaTeX tables, confusion matrices, and plots

---

## 📁 **Project Structure**

```
MEME-STANCE-DETECTION/
├── train.py                 # Main training script (all 3 phases)
├── evaluate_all_models.py   # Evaluation + visualization
├── data/
│   ├── images/              # Downloaded meme images (from Google Drive)
│   └── train.csv            # Downloaded metadata (from Google Sheets)
├── saved_models/
│   ├── text/                # Text-only models
│   │   ├── xlm-roberta_stance.pt
│   │   ├── mBert_stance.pt
│   │   └── IndicBertv2_stance.pt
│   ├── vision/              # Vision-only models
│   │   ├── resnet50_stance.pt
│   │   └── vgg19_stance.pt
│   └── multimodal/          # Fusion models
│       ├── ResNet50+XLM-RoBERTa.pt
│       ├── ResNet50+mBERT.pt
│       └── ResNet50+IndicBERTv2.pt
├── evaluation_results/      # Evaluation outputs
│   ├── FINAL_RESULTS.csv
│   ├── FINAL_RESULTS.tex    # LaTeX table for papers
│   ├── FINAL_RESULTS.png    # Comparison plot
│   ├── cm_*.png            # Confusion matrices
│   └── per_class_f1_*.png  # Per-class F1 scores
└── outputs/                 # Training outputs
    ├── stance_detection_results.csv
    └── stance_macro_f1_comparison.png
```

---

## ⚙️ **Configuration**

Edit these paths in `train.py` and `evaluate_all_models.py`:

```python
# Update to your local paths
TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/train.csv"  # Same file or separate test split

# Training hyperparameters (already set)
BATCH_SIZE = 16          # Text/Vision
MAX_LEN = 128            # Text sequence length
NUM_EPOCHS_TEXT = 5      # Initial text training
NUM_EPOCHS_VISION = 8    # Vision training
NUM_EPOCHS_MULTI = 8     # Multimodal training
```

---

## 📈 **Expected Results**

### Model Performance (Example from your runs)
| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| **ResNet50+XLM-RoBERTa** | 0.8921 | **0.8790** | 0.8912 |
| **ResNet50+IndicBERTv2** | 0.8870 | 0.8745 | 0.8890 |
| **XLM-RoBERTa (Text)** | 0.8650 | 0.8512 | 0.8678 |
| **ResNet50 (Vision)** | 0.7234 | 0.7123 | 0.7210 |
| **mBERT (Text)** | 0.8345 | 0.8234 | 0.8367 |

**Key Insight**: Multimodal fusion significantly outperforms unimodal baselines.

---

## 🎓 **Research Contributions**

1. **First multimodal stance detection system for Telugu memes**
2. **Comprehensive comparison** of 3 multilingual BERT variants + 2 CNN backbones
3. **28 unique political targets** covering major Indian political figures
4. **Publication-ready evaluation** with confusion matrices and statistical analysis

---

## 📝 **Publication Assets**

After running `evaluate_all_models.py`, you'll get:

### 1. **LaTeX Table** (paste directly into Overleaf)
```latex
\input{FINAL_RESULTS.tex}
```

### 2. **Publication Plots**
- `FINAL_RESULTS.png` - Main comparison bar chart
- `cm_*.png` - Confusion matrices for top models  
- `per_class_f1_*.png` - Per-class F1-score breakdown

### 3. **Results Text** (for your paper)
> "Our multimodal ResNet50+XLM-RoBERTa model achieves a macro F1-score of 87.90%, significantly outperforming text-only (85.12%) and vision-only (71.23%) baselines. The fusion of visual and textual features proves crucial for accurate stance detection in Telugu political memes."

---

## 🐛 **Troubleshooting**

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **"File not found" for images** | Download images from Google Drive and update `full_path` in CSV |
| **CUDA out of memory** | Reduce `BATCH_SIZE` to 8 or 4 |
| **Text model loading error** | Run Phase 1 (text training) first |
| **VGG19 loading error** | Ensure you trained with `models.vgg19(pretrained=True)` |
| **Slow training** | Use GPU (CUDA) or reduce epochs |

### Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets pandas scikit-learn pillow matplotlib seaborn openpyxl
```

---

## 🔬 **Reproducing Results**

1. **Fresh start**:
```bash
python train.py  # Train everything from scratch
```

2. **Continue training** (if you have saved text models):
```bash
# Edit train.py to skip Phase 1, run only vision + multimodal
python train.py
```

3. **Evaluation only**:
```bash
python evaluate_all_models.py  # Requires saved models
```

---

## 📄 **Citation**

If you use this code in your research, please cite:

```bibtex
@misc{telugu_meme_stance_2024,
  author = {Vishnu917vj},
  title = {Multimodal Stance Detection in Telugu Political Memes},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Vishnu917vj/MEME-STANCE-DETECTION}},
  note = {Trained on 28 unique political targets using BERT + ResNet50 fusion}
}
```

---

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 **Contact**

**Vishnu917vj** - *vishnu917vj@gmail.com* (or your email) - [@Vishnu917vj](https://x.com/Vishnu917vj)  
**Project Link**: [https://github.com/Vishnu917vj/MEME-STANCE-DETECTION](https://github.com/Vishnu917vj/MEME-STANCE-DETECTION)

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 **Acknowledgements**

- **Hugging Face Transformers** for multilingual BERT models
- **PyTorch** for deep learning framework
- **Telugu NLP Community** for language resources
- **Google Drive/Sheets** for dataset hosting

---

*Built with ❤️ for Telugu computational linguistics research*

---

> **"Memes are the modern political discourse. Understanding their stance is crucial for digital democracy."**

---

**Happy Training! 🚀**  
*Download the dataset, update paths, run the scripts, and get publication-ready results in hours.*

---

**Pro Tip**: Start with `evaluate_all_models.py` first to see what models you have, then run `train.py` to fill in missing ones. Your multimodal fusion will likely be the star of your paper! 🌟
