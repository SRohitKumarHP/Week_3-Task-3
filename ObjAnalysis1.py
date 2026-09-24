import cv2
import numpy as np


# ==========================================
# 1. LOAD IMAGE
# ==========================================
image = cv2.imread("3.jpg")

if image is None:
    print("Error: Image not found!")
    exit()

# Resize while keeping processing manageable
image = cv2.resize(image, (900, 600))

result = image.copy()


# ==========================================
# 2. CONVERT TO GRAYSCALE
# ==========================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ==========================================
# 3. IMPROVE CONTRAST
# CLAHE helps with uneven lighting
# ==========================================
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)


# ==========================================
# 4. REMOVE NOISE
# Bilateral filter removes noise while
# preserving object edges
# ==========================================
denoised = cv2.bilateralFilter(
    enhanced,
    9,
    75,
    75
)


# ==========================================
# 5. SHARPEN THE IMAGE SLIGHTLY
# Helps weak object boundaries
# ==========================================
blur_for_sharp = cv2.GaussianBlur(
    denoised,
    (0, 0),
    3
)

sharpened = cv2.addWeighted(
    denoised,
    1.5,
    blur_for_sharp,
    -0.5,
    0
)


# ==========================================
# 6. AUTOMATIC THRESHOLDING
# Otsu automatically selects the threshold
# ==========================================
_, binary = cv2.threshold(
    sharpened,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)


# ==========================================
# 7. MORPHOLOGICAL CLEANING
# ==========================================
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

# Remove small noise
opened = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

# Fill small gaps and connect broken parts
cleaned = cv2.morphologyEx(
    opened,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)


# ==========================================
# 8. FIND CONTOURS
# ==========================================
contours, hierarchy = cv2.findContours(
    cleaned,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ==========================================
# 9. ANALYZE AND COUNT OBJECTS
# ==========================================
count = 0

image_area = image.shape[0] * image.shape[1]

# Ignore contours smaller than this
min_area = image_area * 0.001


for contour in contours:

    # Calculate actual contour area
    area = cv2.contourArea(contour)

    # Ignore noise and extremely large regions
    if area < min_area:
        continue

    if area > image_area * 0.95:
        continue


    # Valid object
    count += 1


    # --------------------------------------
    # Get bounding box
    # --------------------------------------
    x, y, w, h = cv2.boundingRect(contour)


    # --------------------------------------
    # Aspect ratio
    # --------------------------------------
    aspect_ratio = w / float(h)


    # --------------------------------------
    # Perimeter
    # --------------------------------------
    perimeter = cv2.arcLength(contour, True)


    # --------------------------------------
    # Shape approximation
    # --------------------------------------
    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    vertices = len(approx)


    # --------------------------------------
    # Circularity
    # Helps identify circular objects
    # --------------------------------------
    circularity = 0

    if perimeter > 0:
        circularity = (
            4 * np.pi * area
        ) / (perimeter * perimeter)


    # ======================================
    # CLASSIFY SHAPE
    # ======================================
    if circularity > 0.80:
        shape = "Circular Object"

    elif aspect_ratio > 3 or aspect_ratio < 0.33:
        shape = "Long Object"

    elif vertices == 3:
        shape = "Triangle"

    elif vertices == 4:
        shape = "Rectangle/Square"

    else:
        shape = "Other Object"


    # ======================================
    # DRAW CONTOUR
    # ======================================
    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )


    # # ======================================
    # # DRAW BOUNDING BOX
    # # ======================================
    # cv2.rectangle(
    #     result,
    #     (x, y),
    #     (x + w, y + h),
    #     (255, 255, 255),
    #     2
    # )

    # ======================================
    # DISPLAY SHAPE AND AREA
    # ======================================
    cv2.putText(
        result,
        f"{shape} | Area: {int(area)} px2",
        (x, max(y - 30, 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


    # ======================================
    # DISPLAY WIDTH AND HEIGHT
    # ======================================
    cv2.putText(
        result,
        f"W: {w}px | H: {h}px",
        (x, max(y - 8, 60)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )


cv2.putText(
    result,
    f"Total Objects: {count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    2
)

print("Total Objects:", count)

cv2.imshow("Original", image)
cv2.imshow("Enhanced", enhanced)
cv2.imshow("Denoised", denoised)
cv2.imshow("Binary", binary)
cv2.imshow("Cleaned", cleaned)
cv2.imshow("Final Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()