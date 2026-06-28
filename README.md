# Anime Face Generation using WGAN-GP (PyTorch)

## Project Overview

This project is a deep learning-based Generative Adversarial Network (GAN) built using PyTorch to generate anime-style face images.

The model implements Wasserstein GAN with Gradient Penalty (WGAN-GP), which improves training stability and produces higher quality images compared to traditional GANs.

---

## Problem Statement

Generative Adversarial Networks are powerful models for image generation, but they often suffer from unstable training and mode collapse.

This project addresses these issues by implementing WGAN-GP, which enforces a Lipschitz constraint using gradient penalty, leading to more stable training and improved image quality.

---

## Model Architecture

### Generator

- Takes random noise vector as input
- Uses transposed convolution layers for upsampling
- Batch Normalization and ReLU activations
- Tanh activation for 64x64 RGB image output

### Critic (Discriminator)

- Convolutional neural network architecture
- Outputs real-valued score instead of probability
- Learns Wasserstein distance between real and generated data

---

## Training Strategy

- Wasserstein loss function
- Gradient penalty for enforcing Lipschitz constraint
- Multiple critic updates per generator update (n_critic)
- Adam optimizer used for both networks

---

## Project Structure

Anime-WGAN-GP/
│
├── models/
│ ├── generator.py
│ └── discriminator.py
│
├── utils/
│ ├── dataset.py
│ ├── checkpoint.py
│ ├── gradient_penalty.py
│ └── helpers.py
│
├── outputs/
│ ├── images/
│ └── checkpoints/
│
├── train.py
├── requirements.txt
└── README.md

---

## Results

The model progressively improves over training:

- Early epochs: random noise patterns
- Mid training: blurry face structures
- Later epochs: clearer anime-style faces

Best results are typically observed at intermediate epochs before instability increases.

---

## Training Behavior

GAN training is inherently unstable. Loss values may fluctuate significantly and are not reliable indicators of performance.

Visual inspection of generated images is the primary evaluation method.

---

## Checkpoint System

The project includes checkpoint functionality to:

- Save model state during training
- Resume training from saved checkpoints
- Prevent loss of training progress

---

## Requirements

Install dependencies using:

pip install torch torchvision numpy matplotlib tqdm

---

## How to Run

python train.py

---

## Key Learnings

- Implementation of GANs from scratch using PyTorch
- Understanding instability in adversarial training
- WGAN-GP formulation and gradient penalty implementation
- Balancing generator and critic training
- Model checkpointing and training recovery strategies

---

## Future Improvements

- Add Exponential Moving Average (EMA) generator for stability
- Train higher resolution images (128x128 or above)
- Improve architecture for sharper image generation
- Add real-time training visualization
- Implement FID evaluation for quantitative performance measurement

---

## Author

Vishwa Rajarathne
