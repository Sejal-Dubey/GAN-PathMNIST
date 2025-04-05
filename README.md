# GAN-PathMNIST
# 🧬 GAN Variants for Biomedical Image Synthesis

This repository explores and compares three popular GAN variants — **LSGAN**, **WGAN**, and **WGAN-GP** — trained on the **BloodMNIST** dataset from the MedMNIST collection. The models are evaluated both qualitatively and quantitatively to analyze their ability to generate realistic and diverse medical images.

## 📂 Dataset: BloodMNIST (MedMNIST)

- **Source**: [MedMNIST v2](https://medmnist.com/)
- **Classes**: 8 types of white blood cells
- **Image Size**: 28x28 RGB
- **Samples**: 11,240 images
- **Preprocessing**:
  - Resized to 28x28
  - Normalized to [-1, 1] using `transforms.Normalize([0.5]*3, [0.5]*3)`
  - Loaded using PyTorch `DataLoader` for efficient batching and shuffling


## 🧠 GAN Architecture (DCGAN)

All three GAN models share the same architecture based on **DCGAN**, using:
- `ConvTranspose2d` layers in the **Generator**
- `Conv2d` layers in the **Discriminator**

Architecture is constant across models to isolate the effect of **loss functions**.


## ⚙️ Loss Functions & Training Strategies

### 1. 📘 LSGAN (Least Squares GAN)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (`lr=0.0002`, `betas=(0.5, 0.999)`)
- **Training**:
  - Discriminator minimizes squared difference from real (1) and fake (0)
  - Generator tries to produce images classified as real (1)

### 2. 📙 WGAN (Wasserstein GAN)
- **Loss**: Wasserstein loss
- **Optimizer**: RMSProp (`lr=0.00005`)
- **Training**:
  - Critic (discriminator) outputs unnormalized scores
  - **Weight Clipping** between `[-0.01, 0.01]` to enforce Lipschitz continuity

### 3. 📗 WGAN-GP (Wasserstein GAN with Gradient Penalty)
- **Loss**: Wasserstein loss + Gradient Penalty
- **Optimizer**: Adam (`lr=0.0001`, `betas=(0.0, 0.9)`)
- **Training**:
  - Adds gradient penalty term to enforce Lipschitz constraint
  - Smoother training compared to clipping


## 📊 Quantitative Evaluation Metrics

### 🔢 Inception Score (IS)

- Measures **image quality** and **diversity**
- **Higher** is better

### 📉 Fréchet Inception Distance (FID)

- Measures distance between **real** and **generated** image feature distributions
- **Lower** is better



