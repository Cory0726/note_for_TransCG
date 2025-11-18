import cv2
import numpy as np
from inference import Inferencer

def run_transcg_inference(rgb_path, depth_path, config_path):
    """
    Run TransCG DFNet inference and return refined depth.

    :param rgb_path: (str), Path to RGB image (png / jpg).
    :param depth_path: (str), Path to depth image (png / jpg).
    :param config_path: (str), Path to config file (yaml).

    :return: refined_depth (numpy array, H×W, float32)
    """

    # Load RGB image
    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # Load depth
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise ValueError(f"Failed to read depth image: {depth_path}")

        # auto convert RealSense depth (mm → meter)
        if depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) / 1000.0
        else:
            depth = depth_raw.astype(np.float32)

    # Initialize the inferencer.
    inferencer = Inferencer(cfg_file=config_path)
    # Call inferencer for refined depth
    refined_depth = inferencer.inference(rgb, depth)

    return refined_depth.squeeze()  # H × W

if __name__ == "__main__":
    run_transcg_inference(
        config_path="./configs/inference.yaml"
    )