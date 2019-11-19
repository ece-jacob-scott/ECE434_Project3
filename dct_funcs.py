import scipy.fftpack as sp
import numpy as np


def dct2(mat):
    return sp.dct(sp.dct(mat, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(mat):
    return sp.idct(sp.idct(mat, axis=0, norm='ortho'), axis=1, norm='ortho')


def sudo_blockproc(image, func, block_size=(8, 8)):
    if not callable(func):
        raise Exception("func must be a function")
    [height, width] = image.shape
    empty = np.zeros((height, width))
    for i in np.r_[:height:block_size[0]]:
        for j in np.r_[:width:block_size[1]]:
            empty[i:i+8, j:j+8] = func(image[i:i+8, j:j+8])
    return empty
