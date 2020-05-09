import cv2
import numpy as np

img   =   cv2.imread("123.jpg",0) #gray
#-------------------------------------
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX,0.5,absY,0.5,0)
#-------------------------------------
Gaus = cv2.GaussianBlur(img,(3,3),0)
Canny  = cv2.Canny(Gaus , 50 , 150)
#-------------------------------------
cv2.imshow("original",img)
cv2.imshow("Sobel_X",absX)
cv2.imshow("Sobel_Y",absY)
cv2.imshow("Sobel",Sobel)
cv2.imshow("Canny",Canny)
#-------------------------------------
cv2.imwrite('original.png',img)
cv2.imwrite('Sobel_X.png',absX)
cv2.imwrite('Sobel_Y.png',absY)
cv2.imwrite('Sobel.png',Sobel)
cv2.imwrite('Canny.png',Canny)
cv2.waitKey(0)
