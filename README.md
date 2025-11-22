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

##### 2. Download the dataset

- The proposed framework PICNet is trained by employing the NCT-CRC-HE-100K histopathology dataset, which can be downloaded in [**here**](https://zenodo.org/records/1214456).
- After processing, 24,000 measured intensity images and 5,098 object phase images at a spatial resolution of 224 × 224 pixels are obtained.
- The dataset encompasses diverse biological tissue types, including smooth muscle, normal colon mucosa, and lymphocytes. 

##### 3. Run training code

-  Run the code file "main_train.py". Update the path of dataset file and result saving.

##### 4. Run test code

- Run the code file "main_train_simulation.py" and "main_test_experiment.py"
- The experimental data can be obtained from the authors upon reasonable request.

## Selective Results

##### 1. Quantitative phase imaging of quantitative phase target

We experimentally demonstrate quantitative phase imaging of quantitative phase target.

<p align="left">
<img src="result/QPT.gif", width="500">
<p align="left">

##### 2. Application in biological imaging
We experimentally demonstrate the application of PICNet in biological imaging.

<p align="left">
<img src="result/tissue.gif", width="800">
<p align="left">
  
## Citation

```BibTex
@article{xu2025model,
  title={Adaptive aberration-corrected quantitative phase microscopy via physics-informed deep learning},
  author={Xu, Danlin and Cao, Liangcai, and Sun, Hongbo},
  journal={xxxx},
  volume={xxxx},
  number={xxxx},
  pages={xxxx},
  year={2025},
  publisher={xxxx}
}
```
