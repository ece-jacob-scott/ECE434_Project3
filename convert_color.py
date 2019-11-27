# Y_d=0.257*R+0.504*G+0.098*B+16;
# C_b=-0.148*R-0.291*G+0.439*B+128;
# C_r=0.439*R-0.368*G-0.071*B+128;


def rgb_2_ycrcb(pixel):
    Y = (0.257 * pixel[0]) + (0.504 * pixel[1]) + (0.098 * pixel[2]) + 16
    Cb = -(0.148 * pixel[0]) - (0.291 * pixel[1]) + (0.439 * pixel[2]) + 128
    Cr = (0.439 * pixel[0]) - (0.368 * pixel[1]) - (0.071 * pixel[2]) + 128
    return [Y, Cb, Cr]


def ycrcb_2_rgb(pixel):
    R = (1.164 * pixel[0]) + (0.0 * pixel[1]) + (1.596 * pixel[2])
    G = (1.164 * pixel[0]) - (0.392 * pixel[1]) - (0.813 * pixel[2])
    B = (1.164 * pixel[0]) + (2.017 * pixel[1]) + (0.0 * pixel[2])
    return [R, G, B]
