
# Depth Map Computation using SSD and NCC

This project demonstrates the computation of depth maps using Sum of Squared Differences (SSD) and Normalized Cross-Correlation (NCC) methods.

## Files

- `ex4_main.py`: Contains the main script for reading images, computing depth maps using SSD and NCC, and displaying the results.
- `ex4_utils.py`: Utility functions for disparity computation.

## Requirements

- OpenCV
- NumPy
- Matplotlib


This will display the depth maps computed using SSD and NCC methods.

## Functions

### `disparitySSD(img_l, img_r, disp_range, k_size)`

Computes the disparity map using the Sum of Squared Differences (SSD) method.

- `img_l`: Left image.
- `img_r`: Right image.
- `disp_range`: Tuple indicating the minimum and maximum disparity range.
- `k_size`: Kernel size for computing the SSD.

### `disparityNC(img_l, img_r, disp_range, k_size)`

Computes the disparity map using the Normalized Cross-Correlation (NCC) method.

- `img_l`: Left image.
- `img_r`: Right image.
- `disp_range`: Tuple indicating the minimum and maximum disparity range.
- `k_size`: Kernel size for computing the NCC.

## Example

```python
L = cv2.imread(os.path.join('input', 'pair0-L.png'), cv2.IMREAD_GRAYSCALE) / 255.0
R = cv2.imread(os.path.join('input', 'pair0-R.png'), cv2.IMREAD_GRAYSCALE) / 255.0
displayDepthImage(L, R, (0, 5), method=disparitySSD)
displayDepthImage(L, R, (0, 5), method=disparityNC)
```

This example reads a pair of images, computes the disparity maps using SSD and NCC methods, and displays them.

