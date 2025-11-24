import cv2
import numpy as np


if __name__ == "__main__":
    raw_depth_mm = np.load("test_img/M1_01_raw_depth.npy").astype(np.uint16)
    print(raw_depth_mm.shape, raw_depth_mm.max(), raw_depth_mm.min())

    cv2.imwrite("test_img/M1_01_raw_depth_mm.png", raw_depth_mm)
    check_depth_mm = cv2.imread("test_img/M1_01_raw_depth_mm.png", cv2.IMREAD_UNCHANGED)
    print(check_depth_mm.shape, check_depth_mm.max(), check_depth_mm.min())
