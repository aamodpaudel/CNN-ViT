# Qualia: A Hybrid CNN-ViT Approach for Quantitatively Decoding & Modeling of Pet Sentiments / Instincts from Visual Data

## Domain: CABA & Computational Animal Psychology 

## Overview

**Qualia** is an end-to-end pet sentiment analysis pipeline combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to quantitatively decode and model pet emotions from facial images. Inspired by the needs of veterinary care for domestic animals and the condition of street dogs in South-Asian cities, our project aims to remove the necessity of a pet trainer in households as well. In this repository, we cover the project details such as:

* **Dataset preparation** (download, augmentation, splitting)
* **Hybrid architecture** combining local (ResNet-50) and global (ViT) feature extraction
* **Training** with mixed-precision, scheduling, and performance tracking
* **Evaluation** and visualization of results

## Datasets

We use the **Oxford-IIIT Pet Dataset** (37 breeds, \~7.3 K images) from the Visual Geometry Group at Oxford.

* **Download**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* **Original images**: \~7,368
* **Augmented images**: \~29,000 (via `utilities/augmentation.py`)
* **Split**: 70% train (\~20 K), 15% validation (\~4 K), 15% test (\~4 K) (via `utilities/split.py`)

* The original veterinary classified datasets we used for augmentation can be found in this link: [Original Datasets](https://drive.google.com/drive/folders/1DHReaqAtvxzKNFn5YvrJ9AkvDizpwGg7?usp=sharing)
* The augmented images datasets that was processed to be ready for splitting can be found in this link: [Augmented Datasets](https://drive.google.com/drive/folders/1a6bbYXf-mcMdaAu1hBR83N7OcuKb_TP3?usp=drive_link)
* The final splitted images datasets that was used for ultimate model training can be found in this link: [Split Datasets](https://drive.google.com/drive/folders/1KtGcq8W4jo8LSuuuilLnmuMu_ibaxss0?usp=drive_link)

**The Google Collab Notebook**: [CNN-ViT Notebook](https://colab.research.google.com/drive/15hD_IoFO7B7PHTO4jiTzLMEuil3SpWWp?usp=drive_link)  

## Architecture

The model backbone is **ResNet-50** (without its final two layers) that outputs a feature map of size **7×7×2048**. A 1×1 convolution projects this to **7×7×768**, which is flattened and prepended with a learnable CLS token. The resulting sequence of length 50 (1 CLS + 49 patches) is processed by a 4-layer, 8-head Transformer encoder (d\_model = 768). Finally, the CLS output is fed into a linear classifier over four emotion classes.

Refer to the diagram below for a 3D-style view of the hybrid CNN–ViT flow:

![Hybrid CNN–ViT Architecture](https://github.com/aamodpaudel/Qualia/blob/main/Visualizations_Generated/3d-architecture.jpg)


## Installation

```bash
git clone https://github.com/aamodpaudel/Qualia.git
cd Qualia
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Data Augmentation

```bash
python utilities/augmentation.py \
  --input_dir classified_images \
  --output_dir augmented_data \
  --target_count 7000  # per class
```

### 2. Data Splitting

```bash
python utilities/split.py \
  --input_dir augmented_data \
  --output_dir augmented_data_split
```

### 3. Training & Evaluation

```bash
python train.py \
  --data_dir augmented_data_split \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4
```

* Model saved as `pet_sentiment_model.pth`
* Metrics: `training_metrics.json`
* Plots: `epoch_vs_accuracy.png`, `epoch_vs_loss.png`, `activation_function_gelu.png`

#### Generated Epoch vs Accuracy Plot


```markdown
![Epoch vs Accuracy](epoch_vs_accuracy.png)
```

![Epoch vs Accuracy](https://github.com/aamodpaudel/Qualia/blob/main/Visualizations_Generated/epoch_vs_accuracy.png)

## Results (Hybrid CNN–ViT: Qualia)

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
