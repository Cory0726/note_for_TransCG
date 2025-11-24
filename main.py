import cv2
import numpy as np


if __name__ == "__main__":
    raw_depth = cv2.imread("test_img/M1_01_intensity_adjusted.png", cv2.IMREAD_UNCHANGED)
    print(np.shape(raw_depth), np.max(raw_depth), np.min(raw_depth))