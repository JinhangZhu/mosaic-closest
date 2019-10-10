import numpy as np 
import cv2
from matplotlib import pyplot as plt

path_im = 'messi.jpg'

img = cv2.imread(path_im)

print(type(img))

#   Get the luminance of an image
def get_luminance(source):
    return sum(source.flatten())/len(source.flatten())

#   Get the contrast of an image
def get_contrast(source):
    return (max(source.flatten()) - min(source.flatten()))/(max(source.flatten()) + min(source.flatten()))

print(get_luminance(img))
print(get_contrast(img))

color = ('b','g','r')
for i,col in enumerate(color):
    print(i)

a_test = np.array([[[1, 4, 7]],
 [[2, 5, 8]],
 [[3, 6, 9]]])
print(a_test)
an_array = a_test.flatten()
print(an_array)

hist_1 = np.zeros((256,1,3))
for i,col in enumerate(color):
    hist_1[:, :, i] = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist_1[:, :, i],color = col)
    plt.xlim([0,256])
plt.show()