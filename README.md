Markdown
# Semiconductor Defect Classification — MobileNet (Improved)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![ONNX](https://img.shields.io/badge/ONNX-Inference-blueviolet)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success)

## 📌 Project Overview
This project implements a **Deep Learning-based Defect Classification System** for semiconductor wafers. It uses a customized **MobileNetV2** architecture to identify various manufacturing defects.

The model is optimized for **Edge Deployment** (e.g., Raspberry Pi, Hailo AI accelerators) by prioritizing low latency and high recall, ensuring that critical defects are not missed during inspection.

## 🚀 Key Features
* **Architecture**: MobileNetV2 (Transfer Learning) with modified head.
* **Deployment Ready**: Includes `.onnx` model format for hardware acceleration.
* **Phase-1 Compliant**: Implements strict data leakage checks and class imbalance handling.
* **High Recall Focus**: Optimized to minimize false negatives (missed defects).
* **Data Integrity**: Includes label encoding for consistent class mapping.

## 📂 Dataset & Classes
The model classifies wafer images into **8 distinct categories**:
1.  `bridge`
2.  `clean` (No defect)
3.  `CMP(scratch)`
4.  `cracks`
5.  `LER` (Line Edge Roughness)
6.  `open`
7.  `others`
8.  `vias`

> **Note:** The project handles class imbalance using computed class weights and controlled oversampling.

## 📁 Repository Structure
| File Name | Description |
| :--- | :--- |
| `defect_classification_mobilenet_improved.ipynb` | **Main Notebook**. Contains the full training pipeline, EDA, data leakage checks, evaluation metrics, and export logic. |
| `best_model_mobilenet_improved.pth` | **PyTorch Weights**. The saved model state dict with the highest validation accuracy/recall during training. |
| `defect_classification_mobilenet_improved.onnx` | **Inference Model**. The optimized ONNX graph, ready for deployment on edge devices or non-Python environments. |
| `label_encoder_mobilenet_improved.pkl` | **Label Encoder**. A serialized Scikit-Learn object to decode model predictions (Integers) back to Class Names (Strings). |

## 🛠️ Installation & Requirements
To run this project locally, ensure you have the following dependencies installed:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn onnx
💻 Usage
1. Training & Evaluation
Open the Jupyter Notebook (.ipynb) to view the training process. The notebook covers:

Data Loading: Image preprocessing and augmentation.

Training: Transfer learning with MobileNetV2.

Evaluation: Confusion Matrix, F1-Score, and per-class Recall.

2. Inference (Python Example)
You can use the trained .pth file or the .onnx file for prediction. Below is a snippet for using the ONNX model:

Python
import onnxruntime as ort
import numpy as np
import pickle

# Load Label Encoder
with open('label_encoder_mobilenet_improved.pkl', 'rb') as f:
    le = pickle.load(f)

# Load ONNX Model
session = ort.InferenceSession("defect_classification_mobilenet_improved.onnx")

# Run Inference (assuming 'input_image' is preprocessed)
inputs = {session.get_inputs()[0].name: input_image}
outputs = session.run(None, inputs)
predicted_idx = np.argmax(outputs[0])

print(f"Predicted Defect: {le.inverse_transform([predicted_idx])[0]}")
📊 Results
Model Size: Lightweight (~2-3 MB), suitable for embedded systems.

Priority Metric: Recall (to ensure safety in defect detection).

Leakage Check: Confirmed 0% overlap between training and validation sets.

📜 License
This project is open-source. Please feel free to fork and contribute.