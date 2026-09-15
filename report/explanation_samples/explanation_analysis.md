# Module A: Explainability Samples & Saliency Analysis

This directory contains forensic explanation samples produced via **Grad-CAM** (Gradient-weighted Class Activation Mapping) on the final convolutional layer (`conv_head`) of EfficientNet-B0.

---

## 1. Forensic Mechanism

Grad-CAM computes the gradients of the score for class $c$ (Synthetic) with respect to feature activation map $A^k$ of the convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

A bilateral color overlay is projected back onto the 224×224 image space, mapping regions of high synthetic artifact salience (hot red/yellow) versus natural consistency (cool blue/transparent).

---

## 2. Comparative Analysis

### Sample 1: Authentic Real Image (`sample_real_gradcam.png`)
- **Ground Truth:** Real (CIFAR-10 photograph)
- **Predicted Verdict:** `Likely Real` (97.6% Confidence)
- **Grad-CAM Profile:** Low, diffuse spatial activation. The convolutional kernels detect natural sensor grain and coherent high-frequency edges with no localized synthetic boundary concentration.

### Sample 2: AI-Generated Synthetic Image (`sample_synthetic_gradcam.png`)
- **Ground Truth:** Synthetic (Stable Diffusion v1.4)
- **Predicted Verdict:** `Likely AI-Generated` (96.7% Confidence)
- **Grad-CAM Profile:** Intense, localized focal activation concentrated along object boundary contours and latent diffusion texture smudges.

---

## 3. Forensic Interpretation & Limits

Grad-CAM illustrates **model attention**, not mathematical proof of manipulation. Within SignalScope, visual explanations serve as **investigative leads** for human forensic analysts, cross-verified with:
1. Temperature-calibrated confidence score
2. 5-probe transformation stability analysis
3. EXIF and C2PA cryptographic provenance
