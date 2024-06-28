import os
import numpy as np
from ex4_utils import *
import cv2
import matplotlib.pyplot as plt

def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()

def main():
    ## 1-a
    # Read images
    i = 0
    if i == 0:
        min_r, max_r = 0, 5
    else:
        min_r, max_r = 0, 150
    L = cv2.imread(os.path.join('input', 'pair%d-L.png' % i), cv2.IMREAD_GRAYSCALE) / 255.0
    R = cv2.imread(os.path.join('input', 'pair%d-R.png' % i), cv2.IMREAD_GRAYSCALE) / 255.0
    # Display depth SSD
    displayDepthImage(L, R, (min_r, max_r), method=disparitySSD)
    # Display depth NC
    displayDepthImage(L, R, (min_r, max_r), method=disparityNC)



if __name__ == '__main__':
    main()