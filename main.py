import cv2
import numpy as np

import tools
from inference import Inferencer
def main():
    raw_depth_mm = cv2.imread("test_img/M1_01_raw_depth_mm.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(raw_depth_mm.shape, raw_depth_mm.max(), raw_depth_mm.min())
    # raw_depth_mm = np.load('test_img/G_04_raw_depth.npy')
    # print(raw_depth_mm.shape, raw_depth_mm.max(), raw_depth_mm.min())

    gray_img = cv2.imread("test_img/M1_01_intensity_darken05.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    print(gray_img.shape, gray_img.max(), gray_img.min())
    # color_img = cv2.imread('test_img/G_04_rgb_aligment.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    # print(color_img.shape, color_img.max(), color_img.min())

    # Convert depth unit mm to m
    raw_depth_m = raw_depth_mm / 1000
    print(raw_depth_m.shape, raw_depth_m.max(), raw_depth_m.min())

    # Normalize the grayscale image to 0 - 1
    gray_img = gray_img / np.max(gray_img)
    print(gray_img.shape, gray_img.max(), gray_img.min())
    # color_img = color_img / 255.0
    # print(color_img.shape, color_img.max(), color_img.min())

    # # Create a fake RGB image by grayscale image
    rgb_fake = np.stack([gray_img, gray_img, gray_img], axis=2)  # (H, W, 3)
    print(rgb_fake.shape, rgb_fake.max(), rgb_fake.min())

    # Initialize the inference, specify the configuration file.
    inferencer = Inferencer(cfg_path='configs/inference.yaml')

    # # Call inferencer for refined depth
    depth_refine, depth_ori = inferencer.inference(
        rgb=rgb_fake,
        depth=raw_depth_m,
        target_size=(640, 480),
        depth_coefficient=2.0,
        inpainting=True
    )
    depth_refine = (depth_refine * 1000).astype(np.uint16)
    print(depth_refine.shape, depth_refine.max(), depth_refine.min())
    depth_refine_heatmap = tools.rawdepth_to_heatmap(depth_refine)
    cv2.imwrite('result_img/M1_01d05_depth_refine.png', depth_refine)
    cv2.imwrite('result_img/M1_01d05_depth_refine_heatmap.png', depth_refine_heatmap)

if __name__ == "__main__":
    main()
