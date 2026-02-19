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

## Code References / Acknowledgements

This project was implemented using PyTorch and transfer learning concepts.  
The implementation was developed by referring to official documentation and educational resources, which were adapted to the PlantVillage dataset and project requirements.

Primary references:

1. PyTorch Official Documentation  
   https://pytorch.org/docs/stable/index.html  

2. Torchvision Models Documentation (MobileNet)  
   https://pytorch.org/vision/stable/models.html  

3. PyTorch Transfer Learning Tutorial  
   https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html  

4. PlantVillage Dataset  
   Hughes, D. P., & Salathé, M. (2015).  
   An open access repository of images on plant health to enable the development of mobile disease diagnostics.  
   https://www.kaggle.com/datasets/emmarex/plantdisease  

The code structure and training pipeline were adapted and modified specifically for this project.
