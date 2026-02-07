# Semiconductor Defect Classification — MobileNet (Improved)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![ONNX](https://img.shields.io/badge/ONNX-Inference-blueviolet)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success)

---

## 📌 Project Overview
This project implements a **Deep Learning-based Defect Classification System** for semiconductor wafers. It uses a customized **MobileNetV2** architecture to identify various manufacturing defects.

The model is optimized for **Edge Deployment** (e.g., Raspberry Pi, NXP i.MX, Hailo AI accelerators) by prioritizing **low latency and high recall**, ensuring that critical defects are not missed during inspection.

The complete pipeline includes **data leakage checks, class imbalance handling, model evaluation, and ONNX export**, making it fully **Phase-1 compliant**.

---

## 🚀 Key Features
- **Architecture:** MobileNetV2 (Transfer Learning) with modified classification head  
- **Deployment Ready:** Includes `.onnx` model format for hardware acceleration  
- **Phase-1 Compliant:**  
  - Strict data leakage checks (hash + filename based)  
  - Safe augmentation strategy (no flips or heavy distortions)  
  - Class imbalance handling using class weights  
- **High Recall Focus:** Optimized to minimize false negatives (missed defects)  
- **Data Integrity:** Includes label encoding for consistent class mapping  

---

## 📂 Dataset & Classes
The model classifies wafer SEM images into **8 distinct defect categories**:

1. `bridge`  
2. `clean` (No defect)  
3. `CMP(scratch)`  
4. `cracks`  
5. `LER` (Line Edge Roughness)  
6. `open`  
7. `others`  
8. `vias`  

> **Note:**  
> Class imbalance is handled using **computed class weights** and controlled sampling during training.

---

## 📁 Repository Structure | File Name | Description | | :--- | :--- | | defect_classification_mobilenet_improved.ipynb | **Main Notebook**. Contains the full training pipeline, EDA, data leakage checks, evaluation metrics, and export logic. | | best_model_mobilenet_improved.pth | **PyTorch Weights**. The saved model state dict with the highest validation accuracy/recall during training. | | defect_classification_mobilenet_improved.onnx | **Inference Model**. The optimized ONNX graph, ready for deployment on edge devices or non-Python environments. | | label_encoder_mobilenet_improved.pkl | **Label Encoder**. A serialized Scikit-Learn object to decode model predictions (Integers) back to Class Names (Strings). |

---

## 🛠️ Installation & Requirements
To run this project locally, install the required dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn onnx onnxruntime
💻 Usage
1️⃣ Training & Evaluation
Open the Jupyter Notebook:

defect_classification_mobilenet_improved.ipynb
The notebook includes:

Dataset loading & preprocessing

Data leakage detection

Transfer learning with MobileNetV2

Evaluation metrics (Accuracy, Precision, Recall, F1-Score)

Confusion matrix & overfitting analysis

ONNX export

2️⃣ Inference (ONNX – Python Example)
import onnxruntime as ort
import numpy as np
import pickle

# Load Label Encoder
with open('label_encoder_mobilenet_improved.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load ONNX Model
session = ort.InferenceSession("defect_classification_mobilenet_improved.onnx")

# input_image must be preprocessed to shape [1, 3, 224, 224]
inputs = {session.get_inputs()[0].name: input_image}
outputs = session.run(None, inputs)

predicted_idx = np.argmax(outputs[0])
predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

print(f"Predicted Defect: {predicted_label}")
📊 Results Summary
Model Size: Lightweight (~2–3 MB), suitable for embedded systems

Priority Metric: Recall (to ensure safety in defect detection)

Evaluation Includes:

Accuracy, Precision, Recall, F1-Score

Per-class metrics

Confusion matrix

Data Leakage: Confirmed 0% overlap across train / validation / test sets

🧪 Phase-1 Evaluation Philosophy
Accuracy alone is insufficient for semiconductor inspection

Recall is prioritized to avoid missing real defects

False alarms are acceptable compared to missed defects

Lightweight CNNs generalize better on small SEM datasets

🧭 Next Stage — Phase-2 Scope
Planned improvements for the next phase include:

Larger and more diverse SEM datasets

Improved generalization across defect sub-types

Model compression (INT8 quantization, pruning)

Hardware-specific optimization for edge AI accelerators

Real-time deployment benchmarking on embedded platforms

📜 License
This project is open-source and intended for research and educational purposes.
Feel free to fork, use, and extend the work.
