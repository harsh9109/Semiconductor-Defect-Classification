🔬 Semiconductor Defect Classification
Phase-2 — Frozen ONNX Inference (Edge Deployment Focus)
📌 Phase-2 Objective

Phase-2 focuses strictly on deployment-ready inference using a frozen ONNX model exported from Phase-1.

Unlike Phase-1 (training & experimentation), Phase-2 simulates a real semiconductor inspection deployment scenario with strict constraints:

🔒 Deployment Constraints

❌ No retraining

❌ No weight updates

❌ No architecture modification

❌ No Test-Time Augmentation (TTA)

❌ No heuristic defect detection

❌ No feature engineering tricks

✅ Single deterministic ONNX forward pass

✅ Threshold-based decision logic only

This ensures:

Fair evaluation

Industrial realism

Edge deployment compliance

Deterministic reproducibility

🧠 Model Architecture

Base Model: MobileNetV2
Export Format: ONNX (FP32)
Inference Engine: ONNX Runtime

📚 Reference

Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018.

Why MobileNetV2?

Depthwise separable convolutions

Inverted residual bottlenecks

Lightweight (~10MB ONNX model)

Low latency

Edge-device compatible

Efficient spatial feature extraction

MobileNetV2 provides an optimal balance between:

Accuracy

Memory footprint

Inference speed

Embedded deployment feasibility

🧪 Defect Classes (8 Categories)

The model classifies SEM die-level wafer images into:

bridge

clean

CMP(scratch)

cracks

LER (Line Edge Roughness)

open

others

vias

These represent common structural and micro-pattern defects observed in semiconductor fabrication processes.

⚙ Phase-2 Compliance Matrix
Constraint	Status
No retraining	✅
No weight updates	✅
No architecture change	✅
No TTA	✅
No heuristic logic	✅
Single ONNX forward pass	✅
Threshold-based decision only	✅

The Phase-2 model is exactly the frozen ONNX model exported from Phase-1.

🔄 Inference Pipeline

The inference process strictly follows deployment-compliant steps:

Load frozen ONNX model (ONNX Runtime)

Resize input SEM image → 224 × 224

Apply training-consistent normalization

Perform single forward pass

Apply softmax

Apply fixed confidence threshold

Return predicted class

Important Notes

No CLAHE

No sharpening

No handcrafted features

No multi-pass inference

No augmentation

This ensures:

Deterministic outputs

Deployment realism

Evaluation fairness

Industrial compatibility

📊 Phase-2 Evaluation Results

Overall Accuracy: 43.92%
Macro Precision: 0.5126
Macro Recall: 0.4081
Macro F1-Score: 0.4220

Metric Interpretation

Macro Precision (0.5126) → Model predictions are reasonably confident when predicting specific classes.

Macro Recall (0.4081) → Some defect classes are harder to detect consistently.

Macro F1 (0.4220) → Balanced performance under strict deployment constraints.

Given the frozen model condition and resolution shift, performance reflects realistic deployment behavior.

📈 Performance Observations
✅ Strong Detection Performance

The model performs relatively better on:

cracks

CMP(scratch)

bridge

These defects exhibit larger structural distortions that are effectively captured by convolutional feature hierarchies.

⚠ Fine-Grained Class Degradation

Lower performance observed in:

LER

vias

open

🔬 Root Cause Analysis — Resolution-Domain Shift
Training Resolution:

224 × 224

Phase-2 Test Resolution:

128 × 128 (upscaled to 224 × 224)

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

Therefore:

Performance degradation is attributed to resolution-domain mismatch rather than architectural instability.

This mirrors real-world industrial deployment challenges.

🏭 Semiconductor Inspection Philosophy

In semiconductor manufacturing:

Missing a real defect is more critical than raising a false alarm.

Therefore, recall is prioritized over purely cosmetic accuracy improvement.

Phase-2 was designed to remain compliant rather than artificially boosting metrics.

🚀 Deployment Characteristics
Property	Value
Model Size	~10MB (FP32 ONNX)
Framework	ONNX Runtime
Inference Type	Single-pass deterministic
Preprocessing	Minimal
Edge Deployment	Supported

Suitable for:

Embedded inspection systems

Edge AI accelerators

Industry 4.0 quality control

On-device semiconductor screening

🔧 Engineering Takeaways

Lightweight CNNs remain viable for industrial inspection.

Resolution consistency is critical for fine-grained defect detection.

Deployment constraints significantly influence classification behavior.

Edge AI requires balancing recall, accuracy, and compute limits.

Resolution-domain alignment is essential for micro-defect detection systems.

📚 References

Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018.

ONNX — Open Neural Network Exchange

ONNX Runtime — Cross-platform high-performance inference engine

📜 License

This project is intended for:

Research

Academic evaluation

Educational demonstration

Hackathon submission

Not intended for direct commercial semiconductor fabrication deployment.