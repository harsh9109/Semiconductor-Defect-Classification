🔬 Semiconductor Defect Classification
🚀 Phase-2 — Frozen ONNX Inference (Edge Deployment Focus)
📌 Phase-2 Objective

Phase-2 focuses strictly on deployment-ready inference using a frozen ONNX model exported from Phase-1.

Unlike Phase-1 (training & experimentation), Phase-2 simulates a real industrial semiconductor inspection deployment scenario with strict compliance constraints.

🔒 Deployment Constraints
Constraint	Status
No retraining	✅
No weight updates	✅
No architecture modification	✅
No Test-Time Augmentation (TTA)	✅
No heuristic defect detection	✅
No feature engineering tricks	✅
Single deterministic ONNX forward pass	✅
Threshold-based decision logic only	✅

✔ Ensures fair evaluation
✔ Maintains industrial realism
✔ Fully edge-deployment compliant
✔ Deterministic & reproducible outputs

🧠 Model Architecture

Base Model: MobileNetV2
Export Format: ONNX (FP32)
Inference Engine: ONNX Runtime
Model Size: ~10MB

📚 Reference

Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018.

💡 Why MobileNetV2?

Depthwise separable convolutions

Inverted residual bottlenecks

Lightweight architecture

Low memory footprint

Low latency inference

Edge-device compatible

MobileNetV2 provides an optimal balance between:

✅ Accuracy

✅ Memory efficiency

✅ Inference speed

✅ Embedded deployment feasibility

🧪 Defect Classes (8 Categories)

The system classifies SEM die-level wafer images into:

bridge

clean

CMP(scratch)

cracks

LER (Line Edge Roughness)

open

others

vias

These represent common structural and micro-pattern defects observed in semiconductor fabrication.

🔄 Inference Pipeline

The inference process strictly follows deployment-compliant steps:

1️⃣ Load frozen ONNX model (ONNX Runtime)
2️⃣ Resize SEM image → 224 × 224
3️⃣ Apply training-consistent normalization
4️⃣ Perform single forward pass
5️⃣ Apply softmax
6️⃣ Apply fixed confidence threshold
7️⃣ Return predicted class

🚫 Explicitly Not Used

No CLAHE

No sharpening

No handcrafted features

No multi-pass inference

No augmentation

✔ Deterministic outputs
✔ Evaluation fairness
✔ Deployment realism

📊 Phase-2 Evaluation Results
Metric	Value
Overall Accuracy	43.92%
Macro Precision	0.5126
Macro Recall	0.4081
Macro F1-Score	0.4220
📌 Metric Interpretation

Macro Precision (0.5126)
→ Predictions are reasonably confident when a class is selected.

Macro Recall (0.4081)
→ Fine-grained defect classes are harder to detect consistently under deployment constraints.

Macro F1 (0.4220)
→ Balanced performance considering frozen-model and resolution mismatch conditions.

These results reflect realistic edge deployment performance, not artificially optimized experimental conditions.

📈 Performance Observations
✅ Strong Detection Performance

Better detection observed in:

cracks

CMP(scratch)

bridge

These defects exhibit larger structural distortions, which are effectively captured by convolutional feature hierarchies.

⚠ Fine-Grained Class Degradation

Lower performance observed in:

LER

vias

open

🔬 Root Cause Analysis — Resolution-Domain Shift
Training Resolution
224 × 224

Phase-2 Test Resolution
128 × 128 → Upscaled to 224 × 224

Effects of Upscaling

Interpolation smoothing

Edge attenuation

Reduced micro-contrast

Loss of high-frequency gradients

Fine-grained defects such as:

Line Edge Roughness (LER)

Micro vias

Narrow open circuits

are highly dependent on edge-level micro-structural fidelity.

Lightweight CNNs like MobileNetV2 use depthwise convolutions, which are computationally efficient but sensitive to spatial resolution degradation.

📌 Performance degradation is attributed to resolution-domain mismatch, not architectural instability.

This mirrors real-world industrial deployment challenges.

🏭 Semiconductor Inspection Philosophy

In semiconductor manufacturing:

❗ Missing a real defect is more critical than raising a false alarm.

Therefore, recall is prioritized over cosmetic accuracy improvements.

Phase-2 strictly maintains compliance rather than artificially boosting metrics.

🚀 Deployment Characteristics
Property	Value
Model Size	~10MB (FP32 ONNX)
Framework	ONNX Runtime
Inference Type	Single-pass deterministic
Preprocessing	Minimal
Edge Deployment	Supported
Suitable For:

Embedded inspection systems

Edge AI accelerators

Industry 4.0 quality control pipelines

On-device semiconductor screening

🔧 Engineering Takeaways

Lightweight CNNs remain viable for industrial inspection.

Resolution consistency is critical for fine-grained defect detection.

Deployment constraints significantly influence classification behavior.

Edge AI requires balancing accuracy, recall, and compute efficiency.

Resolution-domain alignment is essential for micro-defect reliability.

📚 References

Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018

ONNX — Open Neural Network Exchange

ONNX Runtime — High-performance inference engine

📜 License

This project is intended for:

Research

Academic evaluation

Educational demonstration

Hackathon submission

Not intended for direct commercial semiconductor fabrication deployment.
