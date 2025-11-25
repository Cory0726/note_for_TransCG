import cv2
import numpy as np

from inference import Inferencer

from tools import rawdepth_to_heatmap
from img_processing import array_info, apply_mask_keep_black

def main():
    raw_depth_mm = np.load('test_img/M1_01_raw_depth.npy')
    print('raw_depth : ' + array_info(raw_depth_mm))

    grayscale_img = cv2.imread('test_img/M1_01_intensity_darken05.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    # grayscale_img = cv2.imread('test_img/M1_01_intensity_image.png', cv2.IMREAD_UNCHANGED)
    print('grayscale_img : ' + array_info(grayscale_img))

    mask = cv2.imread('test_img/M1_01_intensity_darken05_mask.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    print('mask : ' + array_info(mask))

    mask_raw_depth_mm = apply_mask_keep_black(raw_depth_mm, mask)
    print('mask_raw_depth_mm : ' + array_info(mask_raw_depth_mm))
    mask_raw_depth_mm_heatmap = rawdepth_to_heatmap(mask_raw_depth_mm)
    cv2.imshow('mask_raw_depth_mm_heatmap', mask_raw_depth_mm_heatmap)
    cv2.waitKey(0)

    # Convert depth unit mm to m
    # raw_depth_m = raw_depth_m / 1000
    # print(raw_depth_m.shape, raw_depth_m.max(), raw_depth_m.min())

    # Normalize the grayscale image to 0 - 1
    # gray_img = gray_img / np.max(gray_img)
    # print(gray_img.shape, gray_img.max(), gray_img.min())
    # color_img = color_img / 255.0
    # print(color_img.shape, color_img.max(), color_img.min())

    # # Create a fake RGB image by grayscale image
    # rgb_fake = np.stack([gray_img, gray_img, gray_img], axis=2)  # (H, W, 3)
    # print(rgb_fake.shape, rgb_fake.max(), rgb_fake.min())

    # Initialize the inference, specify the configuration file.
    # inferencer = Inferencer(cfg_path='configs/inference.yaml')

    # # Call inferencer for refined depth
    # depth_refine, depth_ori = inferencer.inference(
    #     rgb=rgb_fake,
    #     depth=raw_depth_m,
    #     target_size=(640, 480),
    #     depth_coefficient=10.0,
    #     inpainting=True
    # )
    # raw_depth_m = (raw_depth_m * 1000).astype(np.uint16)
    # print(raw_depth_m.shape, raw_depth_m.max(), raw_depth_m.min())
    # raw_depth_m_heatmap = tools.rawdepth_to_heatmap(raw_depth_m)
    # depth_refine = (depth_refine * 1000).astype(np.uint16)
    # print(depth_refine.shape, depth_refine.max(), depth_refine.min())
    # depth_refine_heatmap = tools.rawdepth_to_heatmap(depth_refine)
    # cv2.imwrite('result_img/M11_1_raw_depth_heatmap.png', raw_depth_m_heatmap)
    # cv2.imwrite('result_img/M1_11_depth_refine.png', depth_refine)
    # cv2.imwrite('result_img/M1_11_depth_refine_heatmap.png', depth_refine_heatmap)

if __name__ == "__main__":
    main()
