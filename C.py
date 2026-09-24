import cv2
import numpy as np

image = cv2.imread("S&S.jpg")

# crop = image[50:500, 70:10000]

if image is None:
    print("Image not found")
    exit()

result = image.copy()

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

blur = cv2.GaussianBlur(
    gray,
    (3, 3),
    3
)

_, binary = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

object_count = 0

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 800:
        continue

    object_count += 1

    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

print("Total number of objects:", object_count)

cv2.putText(
    result,
    f"Objects: {object_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.imshow("Original", image)
cv2.imshow("Cleaned", binary)
cv2.imshow("Contours and Count", result)

cv2.waitKey(0)
cv2.destroyAllWindows()