import cv2
import myimpy.transformation as tf
import numpy as np

img = cv2.imread('mercy_down.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
cv2.imwrite('gray_mercy.jpg',gray,)
print('done!')

gg = cv2.imread('a.jpg',2)
print(gg.shape)