# Qualia: A Hybrid CNN–ViT Approach for Pet Sentiments and Instincts Analysis

## Domain: CABA & Computational Animal Psychology 

## Overview

**Qualia** is an end-to-end pet sentiment analysis pipeline combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to quantitatively decode and model pet emotions from facial images. Inspired by the poster **"Qualia: A Hybrid CNN–ViT Approach for Quantitatively Decoding & Modeling of Pet Sentiments,"** this repository covers:

* **Dataset preparation** (download, augmentation, splitting)
* **Hybrid architecture** combining local (ResNet-50) and global (ViT) feature extraction
* **Training** with mixed-precision, scheduling, and performance tracking
* **Evaluation** and visualization of results

## Datasets

We use the **Oxford-IIIT Pet Dataset** (37 breeds, \~7.3 K images) from the Visual Geometry Group at Oxford.

* **Download**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* **Original images**: \~7,368
* **Augmented images**: \~29,000 (via `scripts/augment.py`)
* **Split**: 70% train (\~20 K), 15% validation (\~4 K), 15% test (\~4 K)

## Architecture

The model backbone is **ResNet-50** (without its final two layers) that outputs a feature map of size **7×7×2048**. A 1×1 convolution projects this to **7×7×768**, which is flattened and prepended with a learnable CLS token. The resulting sequence of length 50 (1 CLS + 49 patches) is processed by a 4-layer, 8-head Transformer encoder (d\_model = 768). Finally, the CLS output is fed into a linear classifier over four emotion classes.

Refer to the diagram below for a 3D-style view of the hybrid CNN–ViT flow:

![Hybrid CNN–ViT Architecture](3d-architecture.jpg)

## Repository Structure

```
├── classified_images/         # Raw labeled pet images
├── augmented_data/            # Augmented images (total ~29 K)
├── augmented_data_split/      # Splits: train/val/test
│   ├── train/ (70%)
│   ├── val/   (15%)
│   └── test/  (15%)
├── scripts/
│   ├── augment.py             # Balance & augment dataset to 7 K per class
│   ├── split_data.py          # Stratified 70/15/15 split
│   └── train.py               # Model training & plotting
├── training_metrics.json      # Epoch-by-epoch metrics
├── epoch_vs_accuracy.png      # Accuracy curves (embedded below)
├── epoch_vs_loss.png          # Loss curves
└── activation_function_gelu.png  # GELU activation plot
```

## Installation

```bash
git clone https://github.com/<username>/qualia-pet-sentiment.git
cd qualia-pet-sentiment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Data Augmentation

```bash
python scripts/augment.py \
  --input_dir classified_images \
  --output_dir augmented_data \
  --target_count 7000  # per class
```

### 2. Data Splitting

```bash
python scripts/split_data.py \
  --input_dir augmented_data \
  --output_dir augmented_data_split
```

### 3. Training & Evaluation

```bash
python scripts/train.py \
  --data_dir augmented_data_split \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4
```

* Model saved as `pet_sentiment_model.pth`
* Metrics: `training_metrics.json`
* Plots: `epoch_vs_accuracy.png`, `epoch_vs_loss.png`, `activation_function_gelu.png`

#### Embedding the Epoch vs Accuracy Plot

To embed the accuracy plot in this README, include the following markdown:

```markdown
![Epoch vs Accuracy](epoch_vs_accuracy.png)
```

![Epoch vs Accuracy](epoch_vs_accuracy.png)

## Results (Hybrid CNN–ViT)

| Split      | Accuracy |
| ---------- | -------- |
| Training   | 0.999    |
| Validation | 0.851    |
| Test       | 0.842    |

## Future Work

* Anatomy-aware few-shot learning for diverse species beyond pets
* Audio–visual multimodal sentiment tracking
* Advanced augmentation strategies and stronger regularization

## Contributing

Contributions welcome! Open issues or submit PRs for features and improvements.

## License

MIT License. See [LICENSE](LICENSE) for details.
