import numpy as np 
from scipy.spatial.distance import euclidean

source1 = np.array([[1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12]])
source2 = np.array([[12,11,10],
            [9,8,7],
            [6,5,4],
            [3,2,1]])
array1 = source1.flatten()
array2 = source2.flatten()

print('array1: ',array1)
print('array2: ',array2)
dist = euclidean(array1, array2)
print('dist: ',dist)