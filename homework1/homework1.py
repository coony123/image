import cv2
import numpy as np
import time
# -----------------------------------------------------
img = cv2.imread('Lenna.jpg');
# -----------------------------------------------------
# cv2.setUseOptimized(False) # AVX
cv2.setUseOptimized(True) # AVX
cv2.useOptimized()
# -----------------------------------------------------
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
tStart = time.time()
# -----------------------------------------------------
for i in range (1000) :
    kernel    = np.ones((7,7),np.uint8)
    # frame_mor = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    frame_mor = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
# -----------------------------------------------------
# -----------------------------------------------------
tEnd = time.time()
sec = tEnd - tStart
print("It cost sec : %f s" % sec)
# -----------------------------------------------------
cv2.imshow('frame_mor', frame_mor)
# cv2.imshow('gray', gray)
# cv2.imshow('img', img)
# -----------------------------------------------------
# cv2.imwrite('img.jpg', img)
# cv2.imwrite('gray.jpg', gray)
# cv2.imwrite('thresh.jpg', thresh)
# cv2.imwrite('frame_mor_close.jpg', frame_mor)

cv2.waitKey(0)
cv2.destroyAllWindows()