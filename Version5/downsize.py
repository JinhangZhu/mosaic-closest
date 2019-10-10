import cv2

img = cv2.imread('mercy_down.jpg')
img_shape = list(img.shape)
resized = cv2.resize(img,None,fx=0.5,fy=0.5)
cv2.imwrite('mercy_down_down.jpg',resized)
print('done!')