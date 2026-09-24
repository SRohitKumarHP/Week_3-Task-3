import cv2
import numpy as np

image = cv2.imread("2.png")

# Resize if necessary
image = cv2.resize(image, (900, 600))

# crop = image[50:500, 50:1000]

result = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.bilateralFilter(image, 11, 75, 75)

binary= cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    3
)

contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

count = 0


for contour in contours:

    area = cv2.contourArea(contour)

    if area < 1000:
        continue

    count += 1

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / float(h)

    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    if aspect_ratio > 3:
        shape = "Long Object"
    else:
        shape = "Other Object"

    cv2.putText(
        result,
        f"{shape} | Area: {int(area)}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


print("Total Objects:", count)

cv2.imshow("Original", image)
cv2.imshow("Binary", binary)
cv2.imshow("Final Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()