import cv2

img = cv2.imread('2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.bilateralFilter(gray, 3, 33, 33)

# rev, th = cv2.threshold(blur, 180, 255, cv2.THRESH_TOZERO_INV)

th = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    5
)

contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('I', th)
cv2.waitKey(0)
cv2.destroyAllWindows()