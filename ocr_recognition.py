"""
Project 4: Image or Text Recognition (Basic)
DecodeLabs - Industrial Training Kit (Batch 2026)

Path Chosen: Path 1 - Optical Character Recognition (OCR)

Goal:
Implement a basic text recognition pipeline using a pre-trained OCR engine
(Google's Tesseract via pytesseract) with proper image pre-processing.

Key Requirements Covered:
- Use a pre-trained model/library (Tesseract OCR engine)
- Perform recognition on a sample input image
- Display recognized output clearly
- Pre-processing: Grayscale conversion + Gaussian Blur + Adaptive (Otsu) Thresholding
- Accuracy benchmarking: report OCR confidence score (target >= 80%)
"""

import cv2
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


IMAGE_PATH = "sample_image.png"
CONFIDENCE_THRESHOLD = 80  # minimum acceptable confidence (%)


def load_image(path):
    """Load the raw input image using OpenCV."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return image


def preprocess_image(image):
    """
    The Logic Skeleton: Systematic Image Pre-Processing
    Step 1: Grayscale Conversion - collapses 3D RGB into a 1D intensity matrix
    Step 2: Gaussian Blur - smooths the image to reduce noise/artifacts
    Step 3: Adaptive Thresholding (Otsu's Method) - forces every pixel to
            pure black or white for maximum contrast
    """
    # Step 1: Grayscale Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur (reduce noise before thresholding)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Adaptive Thresholding using Otsu's Binarization
    # Otsu automatically calculates the optimal threshold cutoff
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return gray, blurred, binary


def run_ocr(processed_image):
    """
    Run Tesseract OCR on the pre-processed image and extract
    both the recognized text and per-word confidence scores.
    """
    # Extract plain text
    extracted_text = pytesseract.image_to_string(processed_image).strip()

    # Extract detailed data (includes confidence per detected word)
    data = pytesseract.image_to_data(processed_image, output_type=Output.DICT)

    # Filter out empty detections and collect valid confidence scores
    confidences = [
        int(conf) for conf, text in zip(data['conf'], data['text'])
        if text.strip() != "" and int(conf) >= 0
    ]

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return extracted_text, avg_confidence


def display_results(extracted_text, avg_confidence):
    """Step: Display the output clearly, with confidence-based validation."""
    print("=" * 55)
    print("STEP: OCR RESULTS")
    print("=" * 55)
    print(f"Recognized Text:\n\"{extracted_text}\"")
    print()
    print(f"Average Confidence Score: {avg_confidence:.2f}%")
    print()

    # The 80% Threshold: The Confidence Filter (Gatekeeper Rule)
    if avg_confidence >= CONFIDENCE_THRESHOLD:
        print(f"✅ PASSED — Confidence meets the {CONFIDENCE_THRESHOLD}% minimum standard.")
    else:
        print(f"⚠️  BELOW THRESHOLD — Confidence is under {CONFIDENCE_THRESHOLD}%.")
        print("   Consider improving image quality or lighting for better accuracy.")


def main():
    print("=" * 55)
    print("STEP 1: LOADING IMAGE")
    print("=" * 55)
    image = load_image(IMAGE_PATH)
    print(f"Image loaded successfully: {IMAGE_PATH}")
    print(f"Dimensions (Height x Width x Depth): {image.shape}")
    print()

    print("=" * 55)
    print("STEP 2: PRE-PROCESSING PIPELINE")
    print("=" * 55)
    gray, blurred, binary = preprocess_image(image)
    print("Applied: Grayscale -> Gaussian Blur -> Otsu Adaptive Thresholding")

    # Save intermediate outputs for visual confirmation
    cv2.imwrite("output_gray.png", gray)
    cv2.imwrite("output_blurred.png", blurred)
    cv2.imwrite("output_binary.png", binary)
    print("Saved pre-processing stages: output_gray.png, output_blurred.png, output_binary.png")
    print()

    print("=" * 55)
    print("STEP 3: RUNNING OCR (TESSERACT)")
    print("=" * 55)
    extracted_text, avg_confidence = run_ocr(binary)

    print()
    display_results(extracted_text, avg_confidence)


if __name__ == "__main__":
    main()
