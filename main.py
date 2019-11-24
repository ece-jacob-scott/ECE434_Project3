# Heavily inspired by https://tinyurl.com/w5h43l4
import cv2 as cv
import numpy as np
from PIL import Image
import os
import scipy as sp
from dct_funcs import dct2, idct2, sudo_blockproc
from zig_zag import zig_zag
from math import ceil, log, floor
from huffman_tables import Y_AC_Table, Y_DC_Table

SCRIPT_DIR = os.getcwd()
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "images")
# NOTE: For some reason this quantization matrix makes the image green?
Q_MATRIX = (
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99),
)

# NOTE: Using this matrix for quantizing makes the picture look good
# Q_MATRIX = [[1] * 8]*8

Q_FACTOR = 1

# Open image using PIL
image = Image.open(os.path.join(IMAGE_FOLDER, "house.tiff"))
# Conver PIL image to cv2
image = np.array(image)
[image_height, image_width, *__] = image.shape
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
        return np.round(mat // q_mat)
    return _helper_


# pre = image_Y_dct[0:8, 0:8]

image_Y_dct = sudo_blockproc(image_Y_dct, quantize(Q_MATRIX, Q_FACTOR))
image_Cr_dct = sudo_blockproc(image_Cr_dct, quantize(Q_MATRIX, Q_FACTOR))
image_Cb_dct = sudo_blockproc(image_Cb_dct, quantize(Q_MATRIX, Q_FACTOR))

# print(f"Q: {image_Y_dct[0:8, 0:8]}\nPre: {pre}")

# Turn in to 1-D array for huffman coding


def sub_zz(mat):
    zz = []
    height = len(mat)
    width = len(mat)
    for i in range(0, height, 8):
        if i + 8 > height:
            break
        for j in range(0, width, 8):
            if j + 8 > width:
                break
            zz.extend(zig_zag(mat[i:i+8, j:j+8]))
    return np.array(zz)


image_Y_dct_zz = sub_zz(image_Y_dct)
image_Cr_dct_zz = sub_zz(image_Cr_dct)
image_Cb_dct_zz = sub_zz(image_Cb_dct)

# Do Huffman Coding for DC
Y_DC = image_Y_dct_zz[::64]
CrCb_DC = image_Cr_dct_zz[::64] + image_Cb_dct_zz[::64]


def calculate_uncompressed(scan):
    total = 0
    for i in scan:
        if i < 0.1:
            total += 1
            continue
        total += ceil(log(i, 2))
    return total


def calculate_dc_values(scan):
    """Go through all the DC values and calculate the difference in each and 
    how many bits that would take to represent
    """
    last = 0
    total = 0
    for i in scan:
        # Find the difference between the last and current DC value
        diff = abs(last - i)
        last = i
        if diff < 0.1:
            continue
        # Add the amount of bits needed to represent difference to total
        category = ceil(log(diff, 2))
        # Use the DC table with the category value to get the compressed
        # huffman code length
        total += Y_DC_Table[category]
    return total


uncompressed_DC = calculate_uncompressed(Y_DC)
compressed_DC = calculate_dc_values(Y_DC)

uncompressed_DC += calculate_uncompressed(CrCb_DC)
compressed_DC += calculate_dc_values(CrCb_DC)

# print(f"Compressed: {compressed_DC}\nUncompressed: {uncompressed_DC}")

# Do Huffman Coding for AC


def break_into_AC(scan):
    AC_values = []
    for index, i in enumerate(scan):
        if index % 64 == 0:
            continue
        AC_values.append(i)
    return AC_values


Y_AC = break_into_AC(image_Y_dct_zz)
CrCb_AC = break_into_AC(image_Cr_dct_zz) + break_into_AC(image_Cb_dct_zz)


def calculate_ac_values(scan):
    zero_count = 0
    total = 0
    EOB = False
    for i in scan:
        # if end of block then just skip all zeroes
        if EOB and i < 0.1:
            continue
        # if hit more than 9 zeroes then it's an EOB
        if zero_count > 9:
            total += Y_AC_Table["EOB"]
            EOB = True
            zero_count = 0
            continue
        EOB = False
        # if value is 0 then increment zero count
        if i < 0.1:
            zero_count += 1
            continue
        # find the code length for the current expression
        exp = f"{zero_count}|{int(i)}"
        # Means i is greater than 10
        if exp not in Y_AC_Table:
            # thus just encode it without huffman
            total += ceil(log(i, 2))
            zero_count = 0
            continue
        total += Y_AC_Table[exp]
        zero_count = 0
    return total


# print(Y_AC)

uncompressed_AC = calculate_uncompressed(Y_AC)
compressed_AC = calculate_ac_values(Y_AC)

uncompressed_AC += calculate_uncompressed(CrCb_AC)
compressed_AC += calculate_ac_values(CrCb_AC)


# print(f"Compressed: {compressed_AC}\nUncompressed: {uncompressed_AC}")

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

cv.imshow("Combine Image", cv.resize(combine_image, (256 * 2, 256 * 2)))
cv.imshow("Original Image", cv.resize(cv.cvtColor(
    image, cv.COLOR_YCrCb2RGB), (256 * 2, 256 * 2)))
cv.waitKey(0)
print(f"Compressed: {compressed_AC + compressed_DC}")
print(f"Uncompressed: {uncompressed_AC + uncompressed_DC}")
