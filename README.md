# Self-Reflective and Continually-Learning AI Assistant for Crop Disease Diagnosis

A research project investigating the reliability of deep-learning-based crop disease classification using prediction confidence, uncertainty, and feature-space familiarity.

## Project Overview

This project explores whether an AI-based crop disease classifier can identify potentially unreliable predictions and flag them for human review.

The work is divided into multiple PBL phases:

- **PBL-II:** Development of the initial classification and explainability pipeline.
- **PBL-III:** Model comparison and experimental evaluation of prediction reliability.
- **PBL-IV:** Planned investigation of verified feedback and continual learning.

**Current status:** PBL-III experimental reliability-assessment prototype.

The current implementation flags potentially unreliable predictions. It does not correct classifications, update model weights using user feedback, or implement continual learning.

## Dataset

| Property | Description |
|---|---|
| Dataset | PlantVillage |
| Selected categories | Pepper, Potato, Tomato |
| Images | 20,638 |
| Classes | 15 |

The dataset is class-imbalanced and is not included in this repository.

## Classification Models

Three ImageNet-pretrained CNN architectures were retrained using new 15-class classifier heads. Pretrained parameters were frozen, and only the final linear classifier was trained.

Three-fold stratified cross-validation was used, with out-of-fold (OOF) predictions generated for evaluation.

| Architecture | Mean OOF Accuracy | Standard Deviation | Errors |
|---|---:|---:|---:|
| MobileNetV3-Small | 93.3279% | 0.1395% | 1,377 |
| ResNet18 | 93.3327% | 0.3949% | 1,376 |
| **EfficientNet-B0** | **94.6555%** | 0.4022% | **1,103** |

EfficientNet-B0 achieved the highest classification performance, improving accuracy by approximately 1.32 percentage points over the other architectures.

## Reliability Assessment

Three signals were investigated:

- **Confidence (C):** One minus the maximum predicted class probability.
- **Entropy (H):** Normalized predictive entropy across the output classes.
- **Familiarity (S):** One minus cosine similarity between an image embedding and its predicted-class prototype.

Each signal represents a different potential indicator of prediction unreliability.

### Main Findings

- Confidence was the strongest standalone signal for ranking incorrect predictions.
- Entropy was highly correlated with confidence (Spearman correlation approximately 0.9986).
- Feature-space familiarity was comparatively weak as a standalone signal.
- Combining the signals did not consistently improve error detection over the confidence baseline.
- Paired bootstrap analysis did not establish a statistically significant improvement in risk-coverage performance.

The findings indicate that additional reflection signals do not automatically improve prediction reliability.

## Explainability

Grad-CAM was used to qualitatively examine selected correct and incorrect predictions, including highly confident misclassifications.

The visualizations are qualitative and do not establish biological validity or lesion-localization accuracy.

![Guided Grad-CAM Contact Sheet](results/explainability/guided_gradcam_contact_sheet.png)

## Selected Research Results

The repository includes selected evaluation artifacts:

**Classification**
- [Cross-validation results](results/classification/cross_validation/)
- [Class-wise performance](results/classification/classwise_analysis/classwise_oof_performance.csv)
- [Top class confusions](results/classification/confusion_analysis/top_class_confusions.csv)

**Reliability**
- [Calibration summary](results/reliability/calibration/evaluation_v3_corrected_error_calibration_summary.csv)
- [Risk-coverage summary](results/reliability/risk_coverage/evaluation_v3_corrected_risk_coverage_summary.csv)
- [Paired bootstrap analysis](results/reliability/risk_coverage/evaluation_v3_corrected_paired_bootstrap.csv)
- [Familiarity analysis](results/reliability/familiarity/corrected_familiarity_summary.csv)

**Experimental metadata**
- [Evaluation manifest](results/reliability/metadata/)

## Limitations

- Evaluation is based on a selected PlantVillage subset, not independently collected field images.
- The class distribution is imbalanced.
- Only the final classifier layer was trained.
- Generalization to unseen diseases, field conditions, and external datasets has not been established.
- Familiarity similarity is not a calibrated probability of correctness.
- Grad-CAM results are qualitative.
- The current system does not learn from user feedback.

## Future Work

Planned research directions include:

- Testing on field-collected crop images.
- External validation and distribution-shift evaluation.
- Improved uncertainty and familiarity estimation.
- Verified feedback and label correction.
- Continual-learning experiments and evaluation of catastrophic forgetting.

## Repository Structure

```text
crop_disease_ml_project/
├── models/
├── notebook/
├── results/
│   ├── classification/
│   ├── reliability/
│   └── explainability/
├── README.md
├── index.html
└── requirements.txt
```

Model checkpoints, prediction arrays, and the original dataset are maintained separately.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks inside the `notebook/` directory. The PlantVillage dataset must be obtained separately.

## Disclaimer

This is an undergraduate research project and an experimental reliability-assessment prototype. Classification performance on PlantVillage should not be interpreted as proof of reliable performance under real-world agricultural conditions.