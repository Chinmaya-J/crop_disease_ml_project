# crop_disease_ml_project
Problem Statement

This project aims to classify plant leaf diseases using image-based deep learning methods.

📊 Dataset

Dataset: PlantVillage

Number of classes: XX

Train/Validation split: 80/20

Image size: 224x224

🧠 Model Architecture

Pretrained MobileNetV3 (ImageNet)

Final layer modified for disease classes

Frozen backbone (transfer learning)

⚙️ Training Details

Loss Function: CrossEntropyLoss

Optimizer: Adam

Epochs: 3

Batch size: 32

📈 Results

Validation Accuracy: 92%

Confusion matrix included in notebook

🚀 Future Work

Grad-CAM interpretability module

Confidence-based reflection mechanism

Continual learning integration
