# note_for_TransCG
My notes for TransCG.

# Reference
- Paper : [TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline](https://ieeexplore.ieee.org/document/9796631)
- Web :
  - [Github - TransCG](https://github.com/Galaxies99/TransCG)
  - [GraspNet - TransCG](https://graspnet.net/transcg)

# Introduction
## TransCG
Include the **dataset** and the proposed **Depth Filler Net (DFNet)** models.

## TransCG Dataset
- TransCG dataset is available on [**official page**](https://graspnet.net/transcg).
- TransCG dataset is the first large-scale real-world dataset for transparent object depth completion and grasping,
- In total, the dataset contains **57,715 RGB-D images** of **51 transparent objects**.
- The **3D mesh model** of the transparent objects are also provided in the dataset.

## System setup
### Package Install
```commandline
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
