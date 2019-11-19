# Heavily inspired by https://tinyurl.com/w5h43l4
import cv2 as cv
import numpy as np
from PIL import Image
import os
import scipy as sp
from dct_funcs import dct2, idct2, sudo_blockproc
from zig_zag import zig_zag
from math import floor

SCRIPT_DIR = os.getcwd()
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "images")
Q_MATRIX = (
    (3, 5, 7, 9, 11, 13, 15, 17),
    (5, 7, 9, 11, 13, 15, 17, 19),
    (7, 9, 11, 13, 15, 17, 19, 21),
    (9, 11, 13, 15, 17, 19, 21, 23),
    (11, 13, 15, 17, 19, 21, 23, 25),
    (13, 15, 17, 19, 21, 23, 25, 27),
    (15, 17, 19, 21, 23, 25, 27, 29),
    (17, 19, 21, 23, 25, 27, 29, 31),
)
Q_FACTOR = 1

# Open image using PIL
image = Image.open(os.path.join(IMAGE_FOLDER, "house.tiff"))
# Conver PIL image to cv2
image = np.array(image)
# For some reason the image is not in the right format
# source: https://tinyurl.com/sdjjjf5
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# Convert image to Y, Cr, Cb
image = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)

# Show image
# ----------------------
# cv.imshow("Image", image)
# cv.waitKey(0)
# cv.destroyWindow("Image")

# Show individual colors
# ----------------------
image_Y = image[:, :, 0].copy()
# cv.imshow("R part", image_r)
# cv.waitKey(0)
# cv.destroyWindow("R part")

image_Cr = image[:, :, 1].copy()
# cv.imshow("G part", image_g)
# cv.waitKey(0)
# cv.destroyWindow("G part")

image_Cb = image[:, :, 2].copy()
# cv.imshow("B part", image_b)
# cv.waitKey(0)
# cv.destroyWindow("B part")

# Perform blockwise dct
image_Y_dct = sudo_blockproc(image_Y, dct2)
image_Cr_dct = sudo_blockproc(image_Cr, dct2)
image_Cb_dct = sudo_blockproc(image_Cb, dct2)

# Show difference between Y part and Y post dct
# cv.imshow("Image Y", image_Y)
# cv.imshow("Image Y dct", image_Y_dct)
# cv.waitKey(0)

# Quantize dct values


def quantize(q_mat, q_factor):
    # create a quantize function that can be sent
    # to sudo_blockproc
    # TODO: fix this division
    def _helper_(mat):
        return floor(mat / (q_mat + q_factor))
    return _helper_


image_Y_dct = sudo_blockproc(image_Y_dct, quantize(Q_MATRIX, Q_FACTOR))


# Turn in to 1-D array for huffman coding
image_Y_dct_zz = zig_zag(image_Y_dct)
image_Cr_dct_zz = zig_zag(image_Cr_dct)
image_Cb_dct_zz = zig_zag(image_Cr_dct)
