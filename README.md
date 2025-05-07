Instance Segmentation for Waste Classification
This project focuses on developing an instance segmentation model tailored for waste classification tasks. Leveraging the Mask R-CNN architecture with a ResNet-50 backbone, the model is fine-tuned on a curated subset of the TACO dataset, targeting three primary waste categories: plastic, landfill, and organic. The project encompasses data preprocessing, model training, evaluation, deployment via an MLOps pipeline, and a web application for real-time inference.

Table of Contents
Overview

Dataset

Model Architecture

Training Setup

Evaluation Metrics

MLOps Pipeline

Web Application

Results

Conclusion


Overview
The primary objective is to create a robust instance segmentation model capable of accurately identifying and classifying waste materials in real-world scenarios. By integrating advanced deep learning techniques with streamlined deployment processes, this project aims to facilitate efficient waste management solutions.

Dataset
Source: TACO (Trash Annotations in Context) dataset.

Selected Categories: Plastic, Landfill, Organic.

Preprocessing:

Removed instances with segmentation masks smaller than 4×4 pixels.

Excluded objects with bounding box visibility below 30%.

Augmentation: Applied geometric and photometric transformations using the Albumentations library to enhance model generalization.

Model Architecture
Base Model: Mask R-CNN with ResNet-50 backbone and Feature Pyramid Network (FPN).

Modifications:

Reinitialized classification and mask heads to accommodate the three target waste categories.

Implementation: Utilized PyTorch's torchvision library for model development.

Training Setup
Input Size: 512×512 pixels.

Optimizer: Stochastic Gradient Descent (SGD) with momentum of 0.9.

Learning Rate Scheduler: OneCycleLR with cosine annealing; max LR = 5e-3.

Loss Functions: Combined classification loss, bounding box regression loss, and binary mask loss.

Epochs: 60.

Batch Size: 4.

Regularization: Weight decay of 5e-4.

Hardware: Training conducted on NVIDIA T4 GPU via Vertex AI Custom Job.
ResearchGate
+1
Semarak Ilmu
+1

Evaluation Metrics
Fixed IoU Threshold (0.5):

Precision, Recall, mean Intersection over Union (mIoU), Class Accuracy.

COCO-style Evaluation:

Mean Average Precision (mAP) computed across IoU thresholds from 0.5 to 0.95 in increments of 0.05.

Observations:

Stable and higher-scoring results at 0.5 IoU threshold.

Performance declined at stricter thresholds (e.g., 0.85+), indicating minor boundary misalignments.

MLOps Pipeline
Version Control: GitHub for codebase management.

CI/CD: Integrated Cloud Build triggers for automated testing and deployment.

Model Registry: Implemented for versioning and tracking model performance.

Deployment: Streamlined process for transitioning models from development to production environments.
Microsoft Learn
Medium
+1
GitHub
+1

Web Application
Framework: Flask.

Functionality:

User-friendly interface for uploading images.

Real-time instance segmentation and waste classification.

Visualization of segmentation masks and class labels.

Deployment: Containerized using Docker for scalability and ease of deployment.
Harness.io
GitHub
+1
Microsoft Learn
+1

Results
Training Metrics:

mIoU: 0.827.

Class Accuracy: 0.820.

Validation Metrics:

mIoU: 0.818.

Class Accuracy: 0.653.

Training Duration: Approximately 7 hours for 60 epochs (~1 hour 10 minutes per 10 epochs) on NVIDIA T4 GPU.

Insights:

High performance on well-separated and clearly visible objects.

Challenges observed in complex backgrounds and overlapping waste materials.
GitHub

Conclusion
This project successfully demonstrates the development and deployment of an instance segmentation model tailored for waste classification. By fine-tuning Mask R-CNN on a curated dataset and integrating an efficient MLOps pipeline, the model achieves high accuracy and robustness in diverse environments. The accompanying web application further enhances accessibility, allowing for real-time waste detection and classification.



