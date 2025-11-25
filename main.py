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
    grayscale_img = cv2.imread('test_img/M1_01_intensity_image.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
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

    # Convert depth unit mm to m
    mask_raw_depth_m = mask_raw_depth_mm / 1000
    print('mask_raw_depth_m : ' + array_info(mask_raw_depth_m))

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
        depth_coefficient=10.0,
        inpainting=True
    )
    print('depth_refine : ' + array_info(depth_refine))
    depth_refine_heatmap = rawdepth_to_heatmap(depth_refine)
    print('depth_refine_heatmap : ' + array_info(depth_refine_heatmap))
    # cv2.imwrite('result_img/M1_01_SegHand_depth_refine_heatmap.png', depth_refine_heatmap)
    cv2.imshow('depth_refine_heatmap', depth_refine_heatmap)
    cv2.waitKey(0)
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
