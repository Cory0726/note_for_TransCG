import cv2
import numpy as np

from inference import Inferencer

from tools import rawdepth_to_heatmap
from img_processing import array_info, apply_mask_keep_black

def main():
    # Raw depth in mm
    raw_depth_mm = np.load('test_img/M1_01_raw_depth.npy')
    print('raw_depth_mm : ' + array_info(raw_depth_mm))
    # Grayscale image
    grayscale_img = cv2.imread('test_img/M1_01_intensity_darken05.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    print('grayscale_img : ' + array_info(grayscale_img))

    # Mask
    mask = cv2.imread('test_img/M1_01_intensity_darken05_mask.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    print('mask : ' + array_info(mask))

    # Seg hand of raw depth data
    mask_raw_depth_mm = apply_mask_keep_black(raw_depth_mm, mask)
    print('mask_raw_depth_mm : ' + array_info(mask_raw_depth_mm))
    mask_raw_depth_mm_heatmap = rawdepth_to_heatmap(mask_raw_depth_mm)
    # cv2.imshow('mask_raw_depth_mm_heatmap', mask_raw_depth_mm_heatmap)
    # cv2.waitKey(0)

    # Convert depth unit mm to m (mask_raw_depth_mm)
    mask_raw_depth_m = mask_raw_depth_mm / 1000
    print('mask_raw_depth_m : ' + array_info(mask_raw_depth_m))

    # Convert depth unit mm to m (raw_depth_mm)
    raw_depth_m = raw_depth_mm / 1000
    print('raw_depth_m : ' + array_info(raw_depth_m))

    # Create a fake RGB image by grayscale image
    fake_rgb = np.stack([grayscale_img, grayscale_img, grayscale_img], axis=2)  # (H, W, 3)
    print('fake_rgb : ' + array_info(fake_rgb))

    # Normalize the fake RGB image to 0 - 1
    norm_fake_rgb = fake_rgb / 255
    print('norm_fake_rgb : ' + array_info(norm_fake_rgb))

    # Initialize the inference, specify the configuration file.
    inferencer = Inferencer(cfg_path='configs/inference.yaml')

    # # Call inferencer for refined depth
    depth_refine, depth_ori = inferencer.inference(
        rgb=norm_fake_rgb,
        depth=mask_raw_depth_m,
        target_size=(640, 480),
        depth_coefficient=5.0,
        inpainting=True,
    )
    print('depth_refine : ' + array_info(depth_refine))
    depth_refine_heatmap = rawdepth_to_heatmap(depth_refine)
    print('depth_refine_heatmap : ' + array_info(depth_refine_heatmap))
    # cv2.imwrite('result_img/M1_01d05_depth_refine_heatmap.png', depth_refine_heatmap)
    cv2.imshow('depth_refine_heatmap', depth_refine_heatmap)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
