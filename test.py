import PIL
import numpy as np


def func(x):
    x[np.argwhere(x != x.max())] = 0
    return x


arr = np.random.random((4,7, 512, 512))

arr = np.apply_along_axis(func, 1, arr)
arr = (arr != 0).astype('float64')
print(arr.sum())
