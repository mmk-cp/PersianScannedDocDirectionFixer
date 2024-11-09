import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import sys

if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
    pass
else:
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image = cv2.imread('input_image.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 1, 10)

# Apply morphological operations to close gaps and improve edge connection
kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.erode(edges, kernel, iterations=1)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

document_contour = None

# Find the contour with 4 corners (representing the document)
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        document_contour = approx
        break

use_result_image = False

# If a document contour is found, perform the perspective transform
if document_contour is not None:
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


    points = document_contour.reshape(4, 2)
    ordered_pts = order_points(points)
    (tl, tr, br, bl) = ordered_pts

    # Determine the width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define destination points and perform the perspective transform
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Save the warped (scanned) document image with a new name
    result_path = 'result_image.jpg'  # Change this to your desired path and name
    cv2.imwrite(result_path, warped)
    use_result_image = True
else:
    print("Document contour could not be found.")

# Load the image
if use_result_image:
    image = cv2.imread('result_image.jpg')
else:
    image = cv2.imread('input_image.jpg')


def rotate_image(image, angle):
    """Rotate the image by the specified angle and prevent cropping."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the new bounding dimensions to avoid cropping
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to account for the translation (move the image so it stays centered)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation with the new dimensions
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def ocr_image(image, lang='fas'):
    """Perform OCR on the image using Tesseract with Persian language."""
    custom_config = r'--oem 3 --psm 6 -l ' + lang
    details = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)
    return details


def get_orientation(image):
    """Determine the correct orientation by rotating and checking OCR confidence."""
    angles = [0, 90, 180, 270]
    ocr_results = []

    # Check only on a portion of the image (center crop) for faster orientation detection
    h, w = image.shape[:2]
    crop_img = image[h // 4:h * 3 // 4, w // 4:w * 3 // 4]  # Crop the center area for faster processing

    for angle in angles:
        rotated_crop = rotate_image(crop_img, angle)
        details = ocr_image(rotated_crop, lang='fas')
        num_chars = sum(len(word) for word in details['text'] if isinstance(word, str))
        ocr_results.append((angle, num_chars))

    # Determine the angle with the most text detected
    best_angle = max(ocr_results, key=lambda x: x[1])[0]
    return best_angle


# Determine the best angle and apply rotation
correct_angle = get_orientation(image)
print(f"Detected best angle: {correct_angle}")

# Rotate the full image to the correct orientation
rotated_image = rotate_image(image, correct_angle)

# Save and display the rotated image
cv2.imwrite('rotated_image.jpg', rotated_image)
