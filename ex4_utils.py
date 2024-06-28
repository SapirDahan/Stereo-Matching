import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    disp_range: Minimum and Maximum disparity range. Ex. (10, 80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1, k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    # Initialize disparity map
    disp_map = np.zeros(img_l.shape)

    # Define the half kernel size
    k_half = k_size // 2

    # Iterate over each pixel in the image
    for r in range(k_half, img_l.shape[0] - k_half):
        for c in range(k_half, img_l.shape[1] - k_half):
            min_ssd = float('inf')
            best_disp = 0

            # Iterate over disparity range
            for d in range(disp_range[0], disp_range[1] + 1):
                if c - d - k_half < 0:
                    continue

                # Define patches
                patch_l = img_l[r - k_half:r + k_half + 1, c - k_half:c + k_half + 1]
                patch_r = img_r[r - k_half:r + k_half + 1, c - k_half - d:c + k_half + 1 - d]

                # Compute SSD
                ssd = np.sum((patch_l - patch_r) ** 2)

                # Find the minimum SSD
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disp = d

            # Store the best disparity
            disp_map[r, c] = best_disp

    return disp_map


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    disp_range: Minimum and Maximum disparity range. Ex. (10, 80)
    k_size: Kernel size for computing the NCC, kernel.shape = (k_size*2+1, k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    # Initialize disparity map
    disp_map = np.zeros(img_l.shape)

    # Define the half kernel size
    k_half = k_size // 2

    # Iterate over each pixel in the image
    for r in range(k_half, img_l.shape[0] - k_half):
        for c in range(k_half, img_l.shape[1] - k_half):
            max_ncc = -1
            best_disp = 0

            # Iterate over disparity range
            for d in range(disp_range[0], disp_range[1] + 1):
                if c - d - k_half < 0:
                    continue

                # Define patches
                patch_l = img_l[r - k_half:r + k_half + 1, c - k_half:c + k_half + 1]
                patch_r = img_r[r - k_half:r + k_half + 1, c - k_half - d:c + k_half + 1 - d]

                # Compute mean of patches
                mean_l = np.mean(patch_l)
                mean_r = np.mean(patch_r)

                # Compute normalized patches
                norm_patch_l = patch_l - mean_l
                norm_patch_r = patch_r - mean_r

                # Compute NCC
                numerator = np.sum(norm_patch_l * norm_patch_r)
                denominator = np.sqrt(np.sum(norm_patch_l ** 2) * np.sum(norm_patch_r ** 2))

                if denominator != 0:
                    ncc = numerator / denominator
                else:
                    ncc = 0

                # Find the maximum NCC
                if ncc > max_ncc:
                    max_ncc = ncc
                    best_disp = d

            # Store the best disparity
            disp_map[r, c] = best_disp

    return disp_map

