# Deep Learning Mini-Project: Narrow ResNet-18 for CIFAR-10 - Team Epoch Explorers

## Project Overview
This project explores the performance and efficiency of a narrow variant of ResNet-18 for CIFAR-10 classification. Traditional deep convolutional neural networks (CNNs) suffer from vanishing gradients and computational inefficiencies when scaled up. ResNets, with their skip connections, mitigate these challenges, enabling the successful training of deep models. In this project, we investigate a compressed version of ResNet-18 by reducing the network width while maintaining depth, aiming to optimize accuracy with fewer parameters.

## Motivation
ResNet-18 has proven effective for image classification, but full-scale models can be computationally expensive. By narrowing the architecture (reducing the number of channels per layer), we test the feasibility of achieving comparable accuracy with fewer parameters, making the model more suitable for resource-constrained environments.

## Key Contributions
- Implemented a **Narrow ResNet-18**, reducing the number of channels by 40%, significantly lowering model complexity.
- Leveraged **AutoAugment** and **Cutout** data augmentation to improve generalization and robustness.
- Employed an optimized **learning rate schedule** (warmup + cosine annealing) for improved convergence.
- Achieved **95.30% test accuracy**, competing with state-of-the-art results while maintaining efficiency.

## Model Architecture
The architecture follows the standard ResNet-18 design with **BasicBlock residual units**, but with reduced filter sizes:

| Layer | Standard ResNet-18 | Narrow ResNet-18 (Our Model) |
|--------|----------------------|-----------------------------|
| Conv1  | 64 filters           | 38 filters                  |
| Stage 1| 64 filters           | 38 filters                  |
| Stage 2| 128 filters          | 77 filters                  |
| Stage 3| 256 filters          | 154 filters                 |
| Stage 4| 512 filters          | 307 filters                 |
| Fully Connected | 512→10    | 307→10                      |

Total parameters: **4.03 million** (56% fewer than standard ResNet-18).

## Dataset
We use the **CIFAR-10 dataset**, consisting of **60,000 images (32x32) across 10 classes**:
- **Training set**: 50,000 images (5,000 per class)
- **Test set**: 10,000 images (1,000 per class)

## Data Preprocessing & Augmentation
To enhance model generalization, we applied:
1. **Random Cropping & Horizontal Flipping** (standard techniques)
2. **AutoAugment (CIFAR-10 Policy)**: Dynamically applies transformations like rotation, color jitter, shearing.
3. **Cutout Augmentation**: Randomly masks a 16×16 patch per image to improve robustness.
4. **Normalization**: Input images are normalized using CIFAR-10 mean and standard deviation.

## Training Strategy
We trained the model using **SGD with Nesterov momentum (0.9)** and **cross-entropy loss with label smoothing (0.1)**. Key hyperparameters:

| Hyperparameter        | Value                |
|----------------------|---------------------|
| Batch Size          | 128                 |
| Optimizer          | SGD + Momentum (0.9) |
| Learning Rate      | 0.1 (initial)        |
| Weight Decay (L2)  | 5e-4                 |
| LR Warmup          | 25 epochs           |
| LR Scheduler       | Cosine Annealing (300 epochs) |
| Dropout Rate       | 0.3 (before FC layer) |

## Learning Rate Schedule
A two-phase schedule was used:
1. **Warmup Phase (0→0.1 over 25 epochs)**: Prevents instability in early training.
2. **Cosine Annealing (gradual decay to 0 over 225 epochs)**: Ensures smooth convergence.

## Results & Observations
### Test Accuracy Comparisons
| Model | Augmentation | Accuracy |
|-------------|-------------------|-----------|
| Narrow ResNet-18 (Baseline) | Random Crop + Flip | 92.50% |
| Narrow ResNet-18 + Cutout | Cutout added | 94.50% |
| Narrow ResNet-18 + AutoAugment + Cutout | Full augmentation | **95.30%** |

### Key Findings
- **Regularization**: AutoAugment + Cutout effectively mitigated overfitting.
- **Model Efficiency**: Despite a **56% parameter reduction**, the model maintained **near-state-of-the-art accuracy**.
- **Training Stability**: Cosine annealing resulted in smoother convergence than step-wise LR decay.

## Conclusion
This study demonstrates that **a narrower ResNet-18 can achieve state-of-the-art performance on CIFAR-10 with appropriate training strategies**. Our findings highlight that **architectural efficiency and advanced augmentation techniques** can compensate for parameter reductions, making deep learning models more practical for deployment in resource-limited environments.

## Repository
Find the full code and implementation details in our GitHub repository:
[GitHub Link](https://github.com/Rudra-Patil/ECE-GY-9143-ResNet-CIFAR10)

## References
1. He, K. et al. (2015) – Deep Residual Learning for Image Recognition.
2. Krizhevsky, A. (2009) – Learning Multiple Layers of Features from Tiny Images (CIFAR-10 dataset).
3. DeVries, T. & Taylor, G. (2017) – Improved Regularization of CNNs with Cutout.
4. Zhang, H. et al. (2017) – mixup: Beyond Empirical Risk Minimization.
5. Yun, S. et al. (2019) – CutMix: Regularization Strategy for Strong Classifiers.
6. Loshchilov, I., Hutter, F. (2017) - SGDR: Stochastic Gradient Descent with Warm Restarts.
