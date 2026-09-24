import cv2
import numpy as np

image = cv2.imread("2.png")

# Resize if necessary
image = cv2.resize(image, (900, 600))

result = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bilateral filter
blur = cv2.bilateralFilter(image, 11, 75, 75)

# Adaptive threshold
binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    3
)

# Find contours
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

count = 0

for contour in contours:

    # Calculate contour area
    area = cv2.contourArea(contour)

    # Ignore small contours
    if area < 1000:
        continue

    count += 1

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate aspect ratio
    aspect_ratio = w / float(h)

    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)

    # Approximate contour shape
    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    # Draw contour
    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    # Draw bounding rectangle
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    # Classify object
    if aspect_ratio > 3:
        shape = "Long Object"
    else:
        shape = "Other Object"

    cv2.putText(
        result,
        f"{shape} | Area: {int(area)} px2",
        (x, y - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        f"Width: {w}px | Height: {h}px",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )
print("Contour Points:")
for point in contour:
    x_point, y_point = point[0]
    print("(", x_point,", ", y_point, ")")

cv2.putText(
    result,
    f"Total no. of Objects: {count}",
    (20, 40),                      
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),                   
    2
)

print("Total Objects:", count)

cv2.imshow("Original", image)
cv2.imshow("Binary", binary)
cv2.imshow("Final Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()