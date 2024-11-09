# Persian Document Scanner with OCR and Automatic Orientation Correction

Description:
This project leverages OpenCV and Tesseract OCR to detect and scan documents from images. It locates document contours,
corrects perspective for a clean, top-down view, and applies OCR (using Persian language support) to recognize and
extract text. The script also detects and adjusts image orientation based on OCR confidence, ensuring proper
readability.

## Requirements:

- **Python 3.6+**
- **OpenCV** for image processing (cv2 module)
- **Pytesseract** for Optical Character Recognition (OCR)

## Python Libraries:

- `numpy` : For numerical operations
- `cv2` : OpenCV library for image processing
- `pytesseract` : Tesseract OCR wrapper

## Steps to Run:

1. Install Python libraries:

  ```bash
  pip install numpy opencv-python pytesseract
  ```

2. Install Tesseract OCR:

- Windows: Download and install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki.
- MacOS/Linux: Install Tesseract via package managers, e.g., `brew install tesseract` (Mac) or
  `sudo apt install tesseract-ocr` (Linux).

## Usage

1. Prepare Input Image: Save the document image as input_image.jpg in the same directory (You can use PNG file).
2. Run the Python script `app.py` using the command:

```bash
python app.py
```

3. Output:

- `result_image.jpg` : Scanned document image after perspective correction.
- `rotated_image.jpg` : Final image with corrected orientation.

## How It Works:

1. Edge Detection and Contour Detection:

-
    - Converts the image to grayscale, applies Gaussian blur, and performs Canny edge detection.
-
    - Identifies document boundaries and applies morphological operations to close gaps.

2. Perspective Correction:

-
    - Locates a contour with four corners (assumed to be the document) and applies perspective transformation to get a
      top-down view.

3. OCR and Orientation Detection:

-
    - Runs OCR on the center portion of the image at various angles (0°, 90°, 180°, and 270°).
-
    - Chooses the angle with the highest OCR confidence for final orientation.

## Notes

- This script is optimized for Persian language OCR; modify the lang parameter in ocr_image if using other languages.
- Ensure `pytesseract.pytesseract.tesseract_cmd` points to your Tesseract installation if running on Windows.

