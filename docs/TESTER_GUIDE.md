# SignalScope — Tester Guide

**SIH 2026 · Internal Hackathon Testing Session**

Thank you for testing SignalScope. This guide explains how to test the system effectively tonight.

---

## What is SignalScope?

SignalScope is a forensic AI tool that analyses images and estimates the probability that they were AI-generated. It uses:
- An EfficientNet-B0 neural network classifier
- Temperature-calibrated probability (not raw confidence)
- 5 robustness probes (how stable is the prediction under compression/resize?)
- Grad-CAM attention maps (where did the model look?)
- EXIF/provenance metadata analysis

**Important:** This is a development baseline. The model was trained on one AI generator (Stable Diffusion v1.4). Results on images from other generators may be inaccurate. Your feedback helps us identify these failures.

---

## Getting Started

1. Open the app URL provided by the team
2. Click the upload area or drag and drop an image
3. Leave the three options checked (Robustness probes, Attention map, Metadata analysis)
4. Click **Analyse Image**
5. Wait 10–30 seconds for results
6. Read the full result — don't just look at the percentage
7. Submit feedback

---

## What to Test

### Category A — Standard cases (baseline verification)

| Image type | Expected | Notes |
|------------|----------|-------|
| Normal photograph (your own phone photo) | Real | Test everyday photos |
| AI art from Midjourney/DALL-E | Synthetic | May fail — out-of-distribution |
| AI art from Stable Diffusion | Synthetic | In-distribution — should work |
| Screenshots | Uncertain | May be misclassified |

### Category B — Edge cases (find failure modes)

Try these specifically:

- **Compressed images** — save a photo as low-quality JPEG and re-upload
- **Resized images** — make an image very small (100×100) or very large
- **Screenshots** — screenshot of a website, app, or document
- **Mixed content** — real photo with AI elements or watermarks
- **Social-media images** — downloaded from Twitter/Instagram (EXIF stripped)
- **Blurry/dark images** — low quality real photos
- **Borderline AI images** — lightly edited or subtle AI generation
- **Black & white images** — real monochrome photographs
- **Medical/scientific images** — X-rays, microscopy, etc.
- **Unusual dimensions** — very wide panoramas or very tall portraits
- **Non-photographic images** — logos, diagrams, illustrations (should flag uncertainty)

### Category C — Stress tests

- Upload the same image 3 times and check if results are consistent
- Upload an image, then re-upload it as a JPEG saved at 50% quality
- Upload a 19 MB image (near the 20 MB limit)
- Upload an invalid file (PDF, text file) — should get a clear error
- Upload an empty file — should get a clear error

### Category D — Mobile testing

- Open the URL on your phone
- Use the camera directly (tap the upload area on mobile)
- Upload from your phone gallery
- Test landscape and portrait images
- Check that the result is readable on a small screen

---

## Reading the Results

### Verdict banner
- **Likely Real** (green ✓) — model estimates this is a real photograph
- **Likely AI-Generated** (red ⚠) — model estimates AI generation
- Values near 50% mean high uncertainty

### Calibrated probability
The percentage shown is temperature-calibrated. It is more reliable than the raw model output but still uncertain, especially for out-of-distribution generators.

### Robustness probes
5 probes run the detector under different image transformations. If all 5 agree with the original verdict, stability is 100%. Low stability means the prediction is fragile.

### Attention map (Grad-CAM)
The heatmap shows where the model focused. Red/yellow = high influence. This explains the model's decision but is NOT proof that highlighted areas are AI-generated.

### Metadata analysis
EXIF data is shown if available. Missing EXIF does NOT mean the image is AI-generated — many real images have EXIF stripped.

### Reliability status
- **Stable** — probes agree, prediction is consistent
- **Unstable** — probes disagree, treat the result with extra caution
- **Insufficient evidence** — not enough data to assess stability

---

## Submitting Feedback

After each analysis, please click one of:

- **✓ Correct** — the verdict matches what you expected
- **✗ Incorrect** — the verdict is clearly wrong
- **? Unsure** — you don't know whether the image is real or AI-generated

You can add an optional comment describing what was unexpected.

**Why this matters:** We use feedback to identify failure patterns and guide tomorrow's model improvement. Your "Incorrect" votes are the most valuable data.

**Note:** Feedback does NOT automatically change the model. It is stored for analysis.

---

## If Something Goes Wrong

| Problem | What to do |
|---------|-----------|
| "Analysis Failed" error | Wait 10 seconds and click Try Again |
| Page takes more than 60 seconds | Refresh and try again |
| Blank page | Check your internet connection |
| Upload fails | Check the file is a JPEG/PNG/WebP/BMP under 20 MB |
| Result looks wrong | Submit feedback as "Incorrect" |

---

## What NOT to Do

- Do NOT upload images of people without their permission
- Do NOT submit personally identifiable information in feedback comments
- Do NOT rely on SignalScope for legal or evidentiary purposes
- Do NOT assume a "Likely Real" verdict is a guarantee

---

## Contact

If the server is down or you find a critical bug, contact the team via the group chat.

---

## Technical Context (for curious testers)

- **Model:** EfficientNet-B0, 5.3M parameters
- **Training data:** CIFAKE (60k real + 60k Stable Diffusion v1.4 synthetic)
- **Baseline test AUC:** 0.9319 (in-distribution only)
- **Calibration:** Temperature T=2.876 fitted on validation set
- **Known limitation:** Not tested on DALL-E, Midjourney, Firefly, or other generators
- **Inference time:** ~5–15 seconds CPU (including 5 probes + Grad-CAM)
