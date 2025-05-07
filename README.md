
# Instance Segmentation for Waste Classification

This project implements an instance segmentation model designed for waste classification using the Mask R-CNN architecture with a ResNet-50 backbone. The model is trained on a curated subset of the TACO dataset, classifying waste into three categories: **plastic**, **landfill**, and **organic**. It includes full data processing, model training, evaluation, MLOps pipeline, and a web app for real-time inference.

---

## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Training Setup](#training-setup)  
- [Evaluation Metrics](#evaluation-metrics)  
- [MLOps Pipeline](#mlops-pipeline)  
- [Web Application](#web-application)  
- [Results](#results)  
- [Conclusion](#conclusion)

---

## Overview

The goal is to build a robust segmentation model that identifies and classifies waste in diverse real-world scenes. The project combines modern deep learning methods with practical deployment tools to create a scalable and efficient solution for automated waste management.

---

## Dataset

- **Source:** [TACO Dataset](https://tacodataset.org/)
- **Selected Classes:** Plastic, Landfill, Organic

### Preprocessing Steps:
- Removed masks smaller than **4×4 pixels**
- Filtered out low-visibility objects (bounding box visibility < 30%)

### Augmentations:
- Applied using **Albumentations**
  - Geometric: flips, rotations, scaling, affine
  - Photometric: brightness, contrast, gamma, hue/saturation

---

## Model Architecture

- **Base Model:** Mask R-CNN with ResNet-50 + FPN  
- **Library:** PyTorch (`torchvision.models.detection`)
- **Customizations:**
  - Classification and mask heads adapted to 3 classes

---

## Training Setup

- **Input size:** 512 × 512  
- **Epochs:** 60  
- **Batch size:** 4  
- **Optimizer:** SGD with 0.9 momentum  
- **Learning Rate:** OneCycleLR (max LR = 5e-3, cosine annealing)  
- **Regularization:** Weight decay = 5e-4  
- **Losses:** Combined classification, bbox regression, and binary mask loss  
- **Device:** Trained on NVIDIA T4 GPU (Vertex AI)

---

## Evaluation Metrics

### 1. Fixed IoU Threshold (0.5)
- Precision
- Recall
- Mean Intersection over Union (mIoU)
- Class Accuracy

### 2. COCO-style Evaluation
- **Mean Average Precision (mAP)** from IoU 0.5 to 0.95 (step 0.05)

**Insight:**  
Best results were achieved at the **0.5 IoU** threshold. Performance dropped at higher thresholds due to minor mask misalignments.

---

## MLOps Pipeline

- **Version Control:** GitHub  
- **CI/CD:** Cloud Build triggers for automated testing and deployment  
- **Model Registry:** Version tracking and evaluation metadata  
- **Deployment:** Containerized deployment pipeline from training to production

---

## Web Application

- **Framework:** Flask  
- **Features:**
  - Upload image and view predictions
  - Live inference with segmentation mask and class labels
- **Deployment:** Dockerized for portability and cloud readiness

---

## Results

### Quantitative Metrics (IoU = 0.5)

| Metric                     | Value         |
|---------------------------|---------------|
| **Training mIoU**         | 0.827         |
| **Validation mIoU**       | 0.818         |
| **Training Accuracy**     | 0.820         |
| **Validation Accuracy**   | 0.653         |
| **Training Time**         | ~7 hours (60 epochs) |
| **Avg. Time per 10 Epochs** | ~1 hr 10 min |

### Qualitative Results

## Results

Below are sample visualizations of model input and output:

<p align="center">
  <img src="./result_1.png" alt="Segmentation Result 1" width="45%" />
  <img src="./result_2.png" alt="Segmentation Result 2" width="45%" />
</p>


---

## Conclusion

This project demonstrates the successful development of a deep learning model for waste classification using instance segmentation. The fine-tuned Mask R-CNN achieved strong accuracy and generalization, making it suitable for deployment in diverse environments. With a robust MLOps pipeline and a functional web application, this system can serve as a foundation for smart waste detection solutions at scale.
