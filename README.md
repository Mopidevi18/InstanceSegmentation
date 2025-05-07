Instance Segmentation for Waste Classification

This project focuses on developing an instance segmentation model tailored for waste classification tasks. Leveraging the Mask R-CNN architecture with a ResNet-50 backbone, the model is fine-tuned on a curated subset of the TACO dataset, targeting three primary waste categories: plastic, landfill, and organic. The pipeline includes data preprocessing, model training, evaluation, deployment via an MLOps pipeline, and a web application for real-time inference.

✨ Overview

The primary objective is to create a robust instance segmentation model capable of accurately identifying and classifying waste materials in real-world scenarios. By integrating advanced deep learning techniques with scalable deployment practices, this project supports efficient and sustainable waste management solutions.

📄 Dataset

Source: TACO Dataset

Selected Categories: Plastic, Landfill, Organic

Preprocessing Steps:

Removed instances with segmentation masks < 4×4 pixels

Excluded objects with <30% bounding box visibility

Applied geometric and photometric augmentations using the Albumentations library

💡 Model Architecture

Base Model: Mask R-CNN with ResNet-50 + Feature Pyramid Network (FPN)

Customization: Reinitialized classification and mask heads for 3 waste classes

Implementation: Built using PyTorch's torchvision library

📊 Training Setup

Input Size: 512×512 px

Optimizer: SGD (momentum=0.9)

Learning Rate Scheduler: OneCycleLR with cosine annealing (max LR: 5e-3)

Losses: Classification + Bounding Box Regression + Binary Mask Loss

Epochs: 60

Batch Size: 4

Regularization: Weight decay = 5e-4

Hardware: Trained on NVIDIA T4 GPU via Vertex AI

🔢 Evaluation Metrics

1. Fixed IoU Threshold (0.5):

Precision, Recall, mIoU, Class Accuracy

2. COCO-style Evaluation:

mAP computed across thresholds (0.5 to 0.95 @ step 0.05)

Observations:

Best performance at IoU=0.5

Decreased performance at higher thresholds due to minor boundary misalignments

🚀 MLOps Pipeline

Version Control: GitHub

CI/CD: Cloud Build for automated testing and deployment

Model Registry: Tracks model versions and evaluation metrics

Deployment: Vertex AI & Docker for development-to-production transition

🚪 Web Application

Framework: Flask

Features:

Upload images for real-time segmentation

Display predictions with masks and confidence scores

Deployment: Docker container for scalability

📊 Results

Training Metrics:

mIoU: 0.827

Class Accuracy: 0.820

Validation Metrics:

mIoU: 0.818

Class Accuracy: 0.653

Training Duration: ~7 hours for 60 epochs (~1h 10min per 10 epochs)

Visual Examples:

Original

Inference Result






Key Insights:

High accuracy for clearly visible waste

Challenges in cluttered scenes or overlapping materials

🔹 Conclusion

This project demonstrates the effectiveness of fine-tuning a Mask R-CNN model for instance segmentation in waste classification. With strong training/validation results and a complete MLOps deployment pipeline, the system offers practical potential for smart cities and environmental monitoring. The interactive web application ensures user-friendly, real-time access to the model.

🔗 References

TACO Dataset

Mask R-CNN Paper

PyTorch torchvision

Albumentations

