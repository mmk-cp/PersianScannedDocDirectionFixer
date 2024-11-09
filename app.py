import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import sys

# Set up Tesseract path for Windows
if sys.platform.startswith('win'):
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

input_image_path = "input_image.png"

# Load the image
image = cv2.imread(input_image_path)

# Convert to grayscale and apply Gaussian blur to reduce noise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection to outline the document
edges = cv2.Canny(blurred, 1, 10)

# Use dilation and erosion to close gaps in edges for better contour detection
kernel = np.ones((5, 5), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours and pick the largest ones, as they are more likely to be the document
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

document_contour = None

# Look for a contour with 4 corners, which suggests a document shape
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        document_contour = approx
        break

# If a document contour is found, correct the perspective to get a top-down view
if document_contour is not None:
    def order_points(pts):
        """Sort points in order: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


    # Arrange points and calculate dimensions for perspective transform
    points = document_contour.reshape(4, 2)
    ordered_pts = order_points(points)
    (tl, tr, br, bl) = ordered_pts

    # Calculate the width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Perform the perspective transformation
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Save the result as the scanned document
    result_path = 'result_image.jpg'
    cv2.imwrite(result_path, warped)
    use_result_image = True
else:
    use_result_image = False
    print("Document contour not found.")

# Load either the warped or original image for the next steps
image = cv2.imread(result_path if use_result_image else input_image_path)


def rotate_image(image, angle):
    """Rotate the image by the given angle without cropping."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Center the rotated image
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def ocr_image(image, lang='fas'):
    """Run OCR on the image using Tesseract in the specified language."""
    custom_config = r'--oem 3 --psm 6 -l ' + lang
    return pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)


def get_orientation(image):
    """Rotate the image in 90-degree steps to find the correct orientation based on OCR text detection."""
    angles = [0, 90, 180, 270]
    ocr_results = []

    # Use the center of the image for faster rotation checks
    h, w = image.shape[:2]
    crop_img = image[h // 4:h * 3 // 4, w // 4:w * 3 // 4]

    for angle in angles:
        rotated_crop = rotate_image(crop_img, angle)
        details = ocr_image(rotated_crop, lang='fas')
        num_chars = sum(len(word) for word in details['text'] if word.isalpha())
        ocr_results.append((angle, num_chars))

    # Pick the angle with the most characters detected
    return max(ocr_results, key=lambda x: x[1])[0]


# Detect the best rotation angle
correct_angle = get_orientation(image)
print(f"Detected best angle: {correct_angle}")

# Rotate the full image to correct orientation
rotated_image = rotate_image(image, correct_angle)

# Save the final rotated image
cv2.imwrite('rotated_image.jpg', rotated_image)
