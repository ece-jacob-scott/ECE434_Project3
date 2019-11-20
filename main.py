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
[image_height, image_width, *_] = image.shape
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


def quantize(q_mat, q_factor=1):
    # create a quantize function that can be sent
    # to sudo_blockproc
    # TODO: add q factor support
    def _helper_(mat):
        return np.round(mat / q_mat)
    return _helper_


image_Y_dct = sudo_blockproc(image_Y_dct, quantize(Q_MATRIX, Q_FACTOR))
image_Cr_dct = sudo_blockproc(image_Cr_dct, quantize(Q_MATRIX, Q_FACTOR))
image_Cb_dct = sudo_blockproc(image_Cb_dct, quantize(Q_MATRIX, Q_FACTOR))

# Turn in to 1-D array for huffman coding
image_Y_dct_zz = zig_zag(image_Y_dct)
image_Cr_dct_zz = zig_zag(image_Cr_dct)
image_Cb_dct_zz = zig_zag(image_Cr_dct)

# Do Huffman Coding

# Unzig zag the 1-D array

# Run idct on all (8 8) blocks
image_Y_idct = sudo_blockproc(image_Y_dct, idct2)
image_Cr_idct = sudo_blockproc(image_Cr_dct, idct2)
image_Cb_idct = sudo_blockproc(image_Cb_dct, idct2)

# Converting to displayable format
image_Y_idct = np.array(np.round(image_Y_idct), dtype=np.uint8)
image_Cr_idct = np.array(np.round(image_Cr_idct), dtype=np.uint8)
image_Cb_idct = np.array(np.round(image_Cb_idct), dtype=np.uint8)

# cv.imshow("Image IDCT", image_Y_idct)
# cv.imshow("Image DCT", image_Y)
# cv.waitKey(0)

# Combine Y Cr Cb
combine_image = np.zeros((image_height, image_width, 3), "uint8")
combine_image[..., 0] = image_Y_idct
combine_image[..., 1] = image_Cr_idct
combine_image[..., 2] = image_Cb_idct
combine_image = np.array(combine_image)
combine_image = cv.cvtColor(combine_image, cv.COLOR_YCR_CB2RGB)

cv.imshow("Combine Image", combine_image)
cv.waitKey(0)
