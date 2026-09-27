# Image or Text Recognition (Basic) – Project 4

A text recognition pipeline that uses Tesseract OCR (Optical Character Recognition) with a complete image pre-processing sequence to accurately extract text from images.

## Overview
This project — "Building the Machine's Optic Nose" — bridges the gap between raw visual data and machine-readable intelligence. It uses a pre-trained OCR engine (Tesseract) combined with classic computer vision pre-processing techniques to reliably extract text from an image, validated by a confidence score.

## Path Chosen: Path 1 – Optical Character Recognition (OCR)

## How It Works (Pipeline)
1. **Load Image** – Read the raw input image using OpenCV
2. **Pre-Processing** (The Logic Skeleton):
   - **Grayscale Conversion** – Collapses the 3D RGB matrix into a 1D intensity matrix
   - **Gaussian Blur** – Smooths the image to reduce noise and artifacts
   - **Adaptive Thresholding (Otsu's Method)** – Automatically calculates the optimal cutoff to convert the image into pure black and white
3. **OCR Extraction** – Run Tesseract on the pre-processed image to extract text
4. **Confidence Validation** – Calculate the average confidence score and validate against an 80% minimum threshold (the Gatekeeper Rule)

## Tech Stack
- Python
- OpenCV (`cv2`)
- pytesseract (Tesseract OCR wrapper)
- Pillow

## Files
- `ocr_recognition.py` – Main OCR pipeline script
- `sample_image.png` – Sample test image containing text

## How to Run
```bash
pip install pytesseract opencv-python pillow
python ocr_recognition.py
```
**Note:** Requires the Tesseract OCR engine to be installed separately (not just the Python library). Windows users can download it from the [UB-Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki).

## Example Output
Recognized Text: "DecodeLabs AI Internship"
Average Confidence Score: 94.33%
✅ PASSED — Confidence meets the 80% minimum standard.

## Key Learnings
- Why pre-processing (grayscale, blur, thresholding) significantly improves OCR accuracy
- How Otsu's method automatically determines the optimal binarization threshold
- The importance of confidence-based validation ("Gatekeeper Rule") rather than blindly trusting model output
- Working with a pre-trained model/library instead of building recognition logic from scratch

Built as part of the DecodeLabs AI Internship Program.
