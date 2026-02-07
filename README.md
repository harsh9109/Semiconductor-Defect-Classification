# 🔬 Semiconductor Defect Classification — MobileNet (Improved)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![ONNX](https://img.shields.io/badge/ONNX-Deployment-blueviolet)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success)
![Model](https://img.shields.io/badge/Model-MobileNetV2-lightgrey)

---

## 📌 Project Overview
This project presents a **Deep Learning–based semiconductor wafer defect classification system** built using a customized **MobileNetV2** architecture.

The primary objective is to **accurately detect manufacturing defects while prioritizing high recall**, ensuring that **critical defects are not missed** during inspection.  
The model is designed to be **lightweight, robust, and deployable on edge devices**, making it suitable for real-world semiconductor inspection pipelines.

The complete workflow — from dataset validation to ONNX deployment — is implemented with **Phase-1 hackathon compliance** in mind.

---

## 🎯 Key Objectives
- Detect multiple semiconductor wafer defects reliably  
- Minimize **false negatives** (missed defects)  
- Maintain **edge-friendly model size and latency**  
- Ensure **data integrity and reproducibility**  

---

## 🚀 Key Features
- **Model Architecture:** MobileNetV2 with transfer learning and a custom classification head  
- **Phase-1 Compliant Pipeline:**  
  - Strict data leakage checks (hash-based & filename-based)  
  - Safe augmentation strategy (no flips or semantic distortions)  
  - Class imbalance handling using computed class weights  
- **High-Recall Optimization:** Designed to catch real defects even at the cost of extra false alarms  
- **Deployment Ready:** ONNX export for non-Python and edge inference  
- **Reproducibility:** Label encoder included for consistent class decoding  

---

## 🧪 Defect Classes
The model classifies SEM wafer images into **8 defect categories**:

1. `bridge`  
2. `clean` *(no defect)*  
3. `CMP(scratch)`  
4. `cracks`  
5. `LER` *(Line Edge Roughness)*  
6. `open`  
7. `others`  
8. `vias`  

> **Class Imbalance Handling:**  
> Balanced training is achieved using **computed class weights**, preventing bias toward majority classes.

---

## 📁 Repository Structure
| File / Folder | Description |
|---------------|-------------|
| `defect_classification_mobilenet_improved.ipynb` | Main notebook: EDA, leakage checks, training, evaluation, and ONNX export |
| `best_model_mobilenet_improved.pth` | Best trained PyTorch model weights |
| `defect_classification_mobilenet_improved.onnx` | Deployment-ready ONNX inference model |
| `label_encoder_mobilenet_improved.pkl` | Label encoder for decoding predicted class indices |
| `dataset/` | Organized dataset with `train / validate / test` splits |
| `README.md` | Project documentation |

---

## 🛠️ Installation & Requirements
Install the required dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn onnx onnxruntime
💻 Usage
1️⃣ Training & Evaluation
Open the notebook below to reproduce training and evaluation:

defect_classification_mobilenet_improved.ipynb
The notebook includes:

Dataset loading & preprocessing

Data leakage detection

Transfer learning with MobileNetV2

Evaluation metrics: Accuracy, Precision, Recall, F1-Score

Confusion matrix & overfitting analysis

ONNX model export

2️⃣ Inference Using ONNX (Python Example)
import onnxruntime as ort
import numpy as np
import pickle

# Load label encoder
with open("label_encoder_mobilenet_improved.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load ONNX model
session = ort.InferenceSession("defect_classification_mobilenet_improved.onnx")

# input_image must be preprocessed to shape [1, 3, 224, 224]
inputs = {session.get_inputs()[0].name: input_image}
outputs = session.run(None, inputs)

predicted_idx = np.argmax(outputs[0])
predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

print(f"Predicted Defect: {predicted_label}")
📊 Results Summary (Phase-1)
- **Model Size:** ~10 MB (FP32 ONNX/PyTorch), suitable for edge deployment; can be reduced further using quantization in Phase-2

Priority Metric: Recall (critical defect detection)

Evaluation Includes:

Accuracy, Precision, Recall, F1-Score

Per-class metrics

Confusion matrix

Data Leakage: ✅ 0% overlap confirmed across train / validation / test splits

🧠 Phase-1 Evaluation Philosophy
Accuracy alone is not sufficient for semiconductor inspection

Recall is prioritized to avoid missing real defects

False alarms are preferable to undetected failures

Lightweight CNNs generalize better on limited SEM datasets

🧭 Next Stage — Phase-2 Roadmap
Planned improvements in Phase-2 include:

Expansion of SEM dataset diversity

Improved generalization across defect sub-types

Model compression (INT8 quantization, pruning)

Hardware-specific optimization for edge AI accelerators

Real-time benchmarking on embedded platforms

📜 License
This project is open-source and intended for research and educational purposes.
You are welcome to fork, use, and extend this work.
