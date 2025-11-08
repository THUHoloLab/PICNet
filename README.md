<div align="center">
<h1> Adaptive aberration-corrected quantitative phase microscopy via physics-informed deep learning </h1>

**[Danlin Xu](https://scholar.google.com/citations?user=Pr1mLHQAAAAJ&hl=zh-CN)**, **[Liangcai Cao](https://scholar.google.com/citations?user=FYYb_-wAAAAJ&hl=en)**, and **[Hongbo Sun](https://scholar.google.com/citations?hl=zh-CN&user=b4vkR2gAAAAJ)**

:school: Department of Precision Instruments, Tsinghua University*


<p align="center">
<img src="result/mainfig.jpg", width='800'>
</p>
</div>

Quantitative phase microscopy (QPM) enables label-free imaging and precise characterization of transparent specimens by measuring phase delay. However, optical aberrations induce wavefront distortions that degrade phase reconstruction accuracy, resolution, and contrast. Existing strategies require diverse measurements or iterative optimization, limiting flexibility for real-time applications. Here, we propose an adaptive aberration-corrected QPM system enabled by a physics-informed cycle-consistent network (PICNet) without prior calibration. By incorporating a learnable physical forward model to approximate the practical image formation and enforcing cycle consistency between object and measurement domains, PICNet can reconstruct the object phase from a single-shot measurement while simultaneously inferring complex aberrations that are difficult to characterize explicitly. Our approach achieves a 60\% improvement in structural similarity compared with uncorrected results. Experiments demonstrate that PICNet enables rapid and high-fidelity phase retrieval across diverse biological samples with enhanced robustness to aberrations. This physically reliable and self-calibrating framework establishes a general paradigm for solving inverse problems across various computational imaging modalities. 

## Requirements

PICNet algorithms are implemented with Python in PyCharm 2024.1.3. Experimental pre- and post-processing codes are written in MATLAB.

- MATLAB R2023b or newer versions
- Python 3.11.4, PyTorch >= 2.3.1
- Platforms: Windows 10 / 11

## Quick Examples

##### 1. Prepare the environment

- Download the necessary packages according to [`requirements.txt`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/requirement.txt).

##### 2. Download the pretrained models

- Download the pretrained models for ViDNet and the baseline networks, which can be found [**here**](https://github.com/THUHoloLab/ViDNet/releases). Then, move the `.pth` files into the corresponding folders in [`models`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/models/). Note that for the baseline networks, FastDVDnet has been modified for grayscale video denoising and with batch normalization layers removed. DRUNet has adopted the original architecture and pretrained model provided by the authors (click [**here**](https://github.com/cszn/DPIR/tree/master) for more details).

##### 3. Download the simulation and experimental dataset

- Follow the instructions [**here**](https://github.com/THUHoloLab/STRIVER-deep/blob/master/data/README.md) to download and prepare the dataset.

##### 4. Run demo codes

- **Quick demonstration with simulated data.** Run [`demo_sim.py`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/demo_sim.py) with default parameters.
- **Demonstration with experimental data.** First run [`demo_exp_probe_recovery.m`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/demo_exp_probe_recovery.m) for TV-regularized blind ptychographic reconstruction to retrieve the probe profile and an initial estimate of the sample field. Then run [`demo_exp.py`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/demo_exp.py) for the deep PnP reconstruction.
- **Experimental comparison.** Run `demo_exp_comparison_eXgY.py` with default parameters, where `X` and `Y` denote the experiment and group indices of the dataset.



## Selective Results

##### 1. Holographic imaging of freely moving organisms

We experimentally demonstrate time-resolved holographic imaging of freely moving organisms based on coded ptychography. The following results show the holographic videos of paramecium and rotifer samples, visualized in the HSL color space.

<p align="left">
<img src="imgs/paramecia_1.gif", width="394"> &nbsp;
<img src="imgs/paramecia_2.gif", width="394">
<p align="left">

<p align="left">
<img src="imgs/rotifers.gif", width="800">
<p align="left">


##### 2. Quantitative comparison with existing PnP priors

The following figure shows the experimental reconstruction of a xylophyta dicotyledon stem section translating at a speed of 5 pixels per frame. Compared with other popular PnP priors including [3DTV](https://github.com/THUHoloLab/STRIVER), [DRUNet](https://github.com/cszn/DPIR), and [FastDVDnet](https://github.com/m-tassano/fastdvdnet), ViDNet maintains the finest spatial textures and the best temporal consistency.

<p align="left">
<img src="imgs/comparison.gif", width='800'>
</p>

The following table summarizes the average amplitude PSNR (dB) under varying sample translation speeds. ViDNet yields competitive performance even when the sample is moving **almost an order of magnitude faster**! :wink:

| Speed (pixel/frame) | 3DTV              | DRUNet + 3DTV | FastDVDnet + 3DTV | ViDNet + 3DTV     |
| :----:              | :----:            | :----:        | :----:            | :----:            |
| 0                   | **21.17 (+0.30)** | 17.74         | 18.26             | 20.87             |
| 1                   | 15.68             | 15.58         | 16.87             | **19.83 (+2.96)** |
| 2                   | 14.56             | 14.58         | 16.01             | **19.33 (+3.32)** |
| 3                   | 14.09             | 14.27         | 15.59             | **19.18 (+3.59)** |
| 4                   | 13.80             | 14.07         | 15.05             | **18.77 (+3.72)** |
| 5                   | 13.61             | 13.93         | 14.49             | **18.44 (+3.95)** |
| 6                   | 13.47             | 13.84         | 14.00             | **17.96 (+3.96)** |
| 7                   | 13.37             | 13.77         | 13.64             | **17.40 (+3.63)** |
| 8                   | 13.30             | 13.65         | 13.37             | **17.00 (+3.35)** |
| 9                   | 13.24             | 13.63         | 13.18             | **16.55 (+2.92)** |


## Citation

```BibTex
@article{gao2025model,
  title={Model-based deep learning enables time-resolved computational microscopy},
  author={Gao, Yunhui and Cao, Liangcai},
  journal={xxxx},
  volume={xxxx},
  number={xxxx},
  pages={xxxx},
  year={2025},
  publisher={xxxx}
}
```
