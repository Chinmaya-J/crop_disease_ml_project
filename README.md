# Self-Reflective AI Assistant for Crop Disease Diagnosis

A research-oriented crop disease classification system investigating whether a vision model can assess the reliability of its own predictions.

## Project Overview

This project investigates a **self-reflective crop disease diagnosis system** built on top of a deep-learning image classifier.

The broader research objective is to develop an AI assistant that can:

1. classify crop diseases from leaf images,
2. assess the reliability of its own predictions,
3. identify predictions that should not be trusted automatically,
4. provide visual evidence for its prediction, and
5. eventually use reflection to support continual learning.

The current implementation corresponds to **PBL-3, Presentation 1**, and focuses specifically on the first working **self-reflection prototype**.

Continual learning, long-term memory, and reflection-driven model adaptation are intentionally deferred to the subsequent project phase.

---

## Current Research Question

### PBL-3

> **Can a vision model assess the reliability of its own prediction using additional model-derived evidence?**

The current prototype investigates this using:

* prediction confidence,
* predictive entropy,
* feature-space familiarity, and
* Grad-CAM-based visual evidence.

The individual components are established techniques. The research focus is their integration into a reflection-oriented reliability assessment layer for crop disease classification.

---

## System Concept

The current conceptual formulation is:

$$
R(x)=g(C,H,S,E)
$$

where:

* \(C\) = prediction confidence,
* \(H\) = predictive entropy,
* \(S\) = feature-space familiarity,
* \(E\) = explanation evidence from Grad-CAM.

The current numerical prototype evaluates:

$$
R(x)=g(C,H,S)
$$

while Grad-CAM is used as qualitative supporting evidence.

The reflection layer can therefore transform:

```text
Image
  ↓
Vision Model
  ↓
Prediction
  ↓
Reliability Assessment
  ↓
Accept / Flag for Review
```

The current implementation is a prototype and should not yet be interpreted as a final calibrated reliability system.

---

## Dataset

**Dataset:** PlantVillage
**Subset:** Pepper, Potato and Tomato
**Images:** 20,638
**Classes:** 15

### Classes

* Pepper Bell Bacterial Spot
* Pepper Bell Healthy
* Potato Early Blight
* Potato Late Blight
* Potato Healthy
* Tomato Bacterial Spot
* Tomato Early Blight
* Tomato Late Blight
* Tomato Leaf Mold
* Tomato Septoria Leaf Spot
* Tomato Spider Mites
* Tomato Target Spot
* Tomato Yellow Leaf Curl Virus
* Tomato Mosaic Virus
* Tomato Healthy

The dataset is **not included in this repository**.

---

## Baseline Model

The classification baseline uses:

* **Architecture:** MobileNetV3-Small
* **Pretraining:** ImageNet
* **Input size:** 224 × 224
* **Transfer learning:** Yes
* **Feature extractor:** Frozen
* **Trainable component:** Final classifier layer

The baseline provides the initial prediction on which the reflection layer operates.

---

## Validation Strategy

The model is evaluated using **3-fold stratified cross-validation**.

Out-of-fold (OOF) predictions are collected so that each image is evaluated using a model that did not use that image for training.

### Current Baseline Results

| Fold          |   Accuracy |
| ------------- | ---------: |
| Fold 1        |     93.28% |
| Fold 2        |     93.43% |
| Fold 3        |     93.09% |
| **Mean**      | **93.27%** |
| **Std. Dev.** |  **0.14%** |

These results establish the current PBL-3 baseline before reflection.

---

## Self-Reflection Prototype

The current prototype evaluates three numerical reliability signals.

### 1. Prediction Confidence

$$
C(x)=\max_k p_k
$$

The maximum predicted class probability provides a direct measure of how concentrated the model's prediction is around its selected class.

---

### 2. Predictive Entropy

$$
H(x)=-\sum_k p_k\log p_k
$$

Entropy considers the complete probability distribution.

* Lower entropy → more concentrated prediction
* Higher entropy → more distributed prediction

Normalized entropy is used in the current implementation.

---

### 3. Feature-Space Familiarity

The model's internal representation is compared with the prototype of the predicted class using cosine similarity:

$$
S(x)=\cos(f(x),\mu_{\hat y})
$$

where:

* \(f(x)\) is the image feature representation,
* \(\mu_{\hat y}\) is the prototype of the predicted class.

This provides a second source of evidence beyond the output probability distribution.

---

## Provisional Reflection Rule

The current prototype uses provisional thresholds:

```text
Confidence < 0.80
OR
Normalized Entropy > 0.20
OR
Familiarity < 0.75
        ↓
Flag prediction for reflection/review
```

These thresholds are **prototype values**, not final experimentally optimized thresholds.

They were selected using the observed OOF signal distributions. Therefore, the resulting prototype reflection statistics are not presented as the final leakage-free evaluation of the method.

Formal threshold selection, calibration, selective prediction, risk-coverage analysis, and comparative experiments are part of the next evaluation phase.

---

## Explainability

Grad-CAM / Guided Grad-CAM is used to visualize regions contributing to the model's prediction.

The explanations are treated as **qualitative evidence** supporting the reflection process.

They are not interpreted as proof of biological correctness because lesion-level ground-truth localization is not available in the current dataset.

---

## Current Project Status

### PBL-2 — Completed

Established:

* PlantVillage dataset subset
* MobileNetV3-Small baseline
* transfer learning
* cross-validation
* Grad-CAM-based explainability

### PBL-3 — Current

Developing:

* prediction reliability assessment
* confidence analysis
* entropy-based uncertainty
* feature-space familiarity
* reflection decision rule
* initial reflection prototype

### PBL-3 — Next

Evaluate whether reflection actually improves decision reliability through:

* threshold selection,
* calibration,
* selective prediction,
* risk-coverage analysis,
* ablation studies,
* baseline vs. reflection comparisons.

### PBL-4 — Future

The longer-term system will investigate:

* reflection-driven continual learning,
* memory of previous cases,
* adaptation to new disease patterns,
* catastrophic forgetting,
* and the integrated reflective AI assistant.

---

## Repository Structure

```text
Crop-Disease-Reflective-AI/
│
├── notebooks/
│   └── plant_disease_reflection_p1.ipynb
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── results/
│   ├── confusion_matrix.png
│   └── reflection_gradcam_examples.png
│
├── README.md
├── references.md
├── requirements.txt
└── .gitignore
```

The PlantVillage dataset and trained model checkpoints are intentionally excluded from the repository.

---

## Running the Notebook

Create the required Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/plant_disease_reflection_p1.ipynb
```

The dataset should be placed under the repository's `data/` directory according to the path expected by the notebook.

PyTorch and Torchvision should be installed as a compatible pair appropriate to the available operating system and CUDA configuration.

---

## Reproducibility

The notebook records the experimental configuration and preserves the outputs from the current prototype run.

The current baseline should therefore be regarded as an experimental reference point rather than a final benchmark.

Future evaluation will introduce stricter reproducibility controls and independent evaluation protocols where required.

---

## Research References

### Primary reliability reference

**Zhou, M., Meng, Y., Xu, M., Abdullah, E. R. M. F., Sabri, A. Q. M., & Rahardja, S. (2026).**

*Towards reliable recognition for plant diseases and weeds by learning soft probability population.*

Engineering Applications of Artificial Intelligence, 167, 113618.

DOI: 10.1016/j.engappai.2025.113618

This work is particularly relevant to the project's reliability-oriented direction because it frames plant-disease recognition around determining whether predictions can be trusted and when intervention is preferable for unknown inputs.

Paper: https://doi.org/10.1016/j.engappai.2025.113618

Public implementation: https://github.com/xml94/SPP4OOD

### Supporting uncertainty reference

**Hernández, S., & López, J. L. (2020).**

*Uncertainty quantification for plant disease detection using Bayesian deep learning.*

Applied Soft Computing, 96, 106597.

DOI: 10.1016/j.asoc.2020.106597

This work provides direct precedent for treating predictive uncertainty as an important consideration in plant disease detection rather than relying solely on classification accuracy.

---

## Technical References

1. **PyTorch Documentation**
   https://pytorch.org/docs/stable/index.html

2. **Torchvision MobileNetV3-Small**
   https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html

3. **PyTorch Transfer Learning Tutorial**
   https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

4. **PlantVillage Dataset**
   Hughes, D. P., & Salathé, M. (2015).
   *An open access repository of images on plant health to enable the development of mobile disease diagnostics.*

5. **Grad-CAM**
   Selvaraju, R. R., et al. (2017).
   *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
   ICCV 2017.
   https://arxiv.org/abs/1610.02391

6. **pytorch-grad-cam**
   https://github.com/jacobgil/pytorch-grad-cam

---

## Acknowledgements

This project was developed using PyTorch, Torchvision, and open research resources.

The implementation and experiments were developed and adapted for the project's research objectives and the selected PlantVillage subset.

Model training and experimentation were performed using GPU resources provided by **Manipal University Jaipur**.

---

## Research Disclaimer

This repository represents an ongoing undergraduate research project.

The current self-reflection mechanism is a **prototype**, not a clinically or agriculturally validated diagnostic system.

High classification accuracy on PlantVillage should not be interpreted as equivalent to reliable performance under real-world field conditions. Further evaluation under distribution shift, calibration, selective prediction, and independent datasets is required before practical deployment.
