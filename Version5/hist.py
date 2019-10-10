import cv2 
import myimpy.transformation as tf 
import matplotlib.pyplot as plt

img1 = cv2.imread('images/whiteb.jpg')
img2 = cv2.imread('')

hist1 = tf.calchist(img1)
hist2 = tf.calchist(img2)

plt.figure()
plt.subplot(1,2,1)
plt.bar(hist1)
plt.title('img1')
plt.subplot(1,2,2)
plt.bar(hist2)
plt.title('img2')
plt.show()