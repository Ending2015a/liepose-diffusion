<div align="center">

# Confronting Ambiguity in 6D Object Pose Estimation via Score-Based Diffusion on SE(3)

CVPR 2024

<font size="4">
Tsu-Ching Hsiao&emsp;
Hao-Wei Chen&emsp;
Hsuan-Kung Yang&emsp;
Chun-Yi Lee&emsp;
</font>
<br>

<font size="4">
Elsa Lab, National Tsing Hua University
</font>

| <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Hsiao_Confronting_Ambiguity_in_6D_Object_Pose_Estimation_via_Score-Based_Diffusion_CVPR_2024_paper.pdf">Paper</a> | <a href="https://arxiv.org/abs/2305.15873">arXiv</a> | <a href="https://ending2015a.github.io/liepose-diffusion-page/">Project</a> |







![se3-diffusion-cover-gt-right-1](https://github.com/Ending2015a/liepose-diffusion/assets/18180004/b10445f3-497b-4a84-bc3e-1d7332e1f8b6)


Official re-implementation in [JAX](https://github.com/google/jax).

</div>




## Abstract

Addressing pose ambiguity in 6D object pose estimation from single RGB images presents a significant challenge, particularly due to object symmetries or occlusions. In response, we introduce a novel score-based diffusion method applied to the SE(3) group, marking the first application of diffusion models to SE(3) within the image domain, specifically tailored for pose estimation tasks. Extensive evaluations demonstrate the method's efficacy in handling pose ambiguity, mitigating perspective-induced ambiguity, and showcasing the robustness of our surrogate Stein score formulation on SE(3). This formulation not only improves the convergence of denoising process but also enhances computational efficiency. Thus, we pioneer a promising strategy for 6D object pose estimation.


## Updates

* 2024/05/14: Code released.

## Videos



https://github.com/Ending2015a/liepose-diffusion/assets/18180004/ab0df460-8c73-4b96-9d5a-8f8132d74b37



<details>
  <summary>Click here to see the SYMSOL-T demos</summary>

https://github.com/Ending2015a/liepose-diffusion/assets/18180004/884f22a0-08e6-4f50-b8aa-0d04223444c8


https://github.com/Ending2015a/liepose-diffusion/assets/18180004/8eb7d957-18a7-4dbd-9673-66a6eb14aa77


https://github.com/Ending2015a/liepose-diffusion/assets/18180004/7fb231ac-e73d-4b4c-a241-1047a479b12c


https://github.com/Ending2015a/liepose-diffusion/assets/18180004/ecf578a9-25bf-4d6f-b3a3-c5846d49f858


https://github.com/Ending2015a/liepose-diffusion/assets/18180004/24056e51-52b1-4ba9-8bb9-50cc55febdba

</details>

---
**Table of Contents**

- [Installation](#installation)
    - [Requirements](#requirements)
    - [Setup](#setup)
- [Experiments](#experiments)
    - [SYMSOL](#symsol)
    - [SYMSOL-T](#symsol-t)
    - [T-LESS](#t-less)
- [Metrics](#metrics)
    - [SYMSOL and SYMSOL-T](#symsol-and-symsol-t)
- [Datasets](#datasets)
- [Multi-GPUs Training](#multi-gpus-training)
- [Citation](#citation)


## Installation

### Requirements

Ensure your system meets the following requirements:
- Linux (only tested on Ubuntu 20.04)
- nvidia-docker
- CUDA 12.2 or higher

### Setup

1. Clone this repo with the following command:
```
git clone git@github.com:Ending2015a/liepose-diffusion.git
```

2. Download datasets.
This will download the [TLESS dataset](https://bop.felk.cvut.cz/datasets/#T-LESS) and [VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
```
cd liepose-diffusion
make download
```
*NOTE* if the datasets do not download correctly, you can download them from the links provided in [Datasets](#datasets) section.

3. Build the docker image and start the container
```
make build
make run
# inside the docker
cd /workspace
```

4. Now you are ready to run the experiments. See [Experiments](#experiments).

## Experiments

### SYMSOL

SO(3)
```
python3 -m liepose.exp.symsol.run
```

The result is located at `logs/experiments/symsol-score-flat/.../inference_400000/summary.json`.


### SYMSOL-T

SE(3)
```
python3 -m liepose.exp.symsolt.run "lie_type=[se3]"
```

R3SO(3)
```
python3 -m liepose.exp.symsolt.run "lie_type=[r3so3]"
```

The result is located at `logs/experiments/symsolt-score-flat/.../inference_800000/summary.json`. The `...` depends on the given parameters, e.g. `lie_type=se3+repr_type=tan+...`.

### T-LESS

SE(3)
```
python3 -m liepose.exp.bop.run "lie_type=[se3]"
```

R3SO(3)
```
python3 -m liepose.exp.bop.run "lie_type=[r3so3]"
```

The result is located at `logs/experiments/bop-tless-score-flat/.../inference_400000/summary.json`. The `...` depends on the given parameters, e.g. `lie_type=se3+repr_type=tan+...`.


## Metrics

### SYMSOL and SYMSOL-T

In `summary.json`, you will see the format like
```
{
  "final_metrics": {
    "rot": 0.007605713326483965,
    "rot(deg)": 0.4357752501964569,
    "rot_2": 99.59599999999999,
    "rot_5": 99.88,
    "rot_10": 99.94,
    "rot_id0": 0.007878238335251808,
    "rot(deg)_id0": 0.45138978958129883,
    "rot_2_id0": 99.98,
    "rot_5_id0": 100.0,
    ...
  },
  ...
}
```

The meaning and the shapes' ID is listed as follows:

|Metrics|Meaning|
|-|-|
|`rot`| average rotation errors in radians|
|`rot(deg)`| average rotation errors in degrees|
|`rot_2`| the percentage (%) of the samples rotation errors less than 2 degrees|
|`rot_5`| the percentage (%) of the samples rotation errors less than 5 degrees|
|`rot_10`| the percentage (%) of the samples rotation errors less than 10 degrees|
|`tran`| average translation errors (distance) |
|`tran_0.02`| the percentage (%) of the samples translation errors less than 0.02|
|`tran_0.05`| the percentage (%) of the samples translation errors less than 0.05|
|`tran_0.1`| the percentage (%) of the samples translation errors less than 0.1|
|`add`| average distance of two point clouds (ADD) |
|`add_0.02`| the percentage (%) of the samples average distance less than 0.02|
|`add_0.05`| the percentage (%) of the samples average distance less than 0.05|
|`add_0.1`| the percentage (%) of the samples average distance less than 0.1|
|`geo`| average geodesic distance on SE(3) |
|`geo_0.02`| the percentage (%) of the samples geodesic distance less than 0.02|
|`geo_0.05`| the percentage (%) of the samples geodesic distance less than 0.05|
|`geo_0.1`| the percentage (%) of the samples geodesic distance less than 0.1|
|`{metric}_id*`| the metrics for each shape, e.g. `rot_id0`, `tran_id2`, ...|

|ID|Shape|
|-|-|
|0|tetrahedron|
|1|cube|
|2|icosahedron|
|3|cone|
|4|cylinder|

## Datasets

We use the following datasets in our experiments:

||Download|
|-|:-:|
| [SYMSOL](https://www.tensorflow.org/datasets/catalog/symmetric_solids) | tfds<sup>*1</sup> |
| SYMSOL-T |[gdrive](https://drive.google.com/drive/folders/1bh_ADsPFFWkaj-rY2rjBj9zXBNKSoRDW?usp=drive_link) <sup>*2</sup>|
| [T-LESS](https://bop.felk.cvut.cz/datasets/#T-LESS) |[gdrive](https://drive.google.com/drive/folders/1KvLI0u6tjaNnNv3V6k9AuvuPDCv0WwVq?usp=drive_link)|
| [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) |[gdrive](https://drive.google.com/drive/folders/1neVkV-EVzDIZblxy5svIQtCzf07vkiyB?usp=drive_link)|

\*1 SYMSOL is downloaded automatically via [`tensorflow-datasets`](https://www.tensorflow.org/datasets) during the first run.

\*2 We also provide the scripts for synthesizing SYMSOL-T dataset.
```
# You need to enable the screen/display before `make run`
export DISPLAY=:0
make run

# Make your own SYMSOL-T (25000 samples) = (5 shapes) * (5k per shape)
python3 liepose.data.symsolt.synth --path "dataset/symsolt/my-symsolt-5k" "num_samples=25000"
```
For more configs, see the script.

## Multi-GPUs Training

In default, single-GPU is used for training as we set the `CUDA_VISIBLE_DEVICES='0'` inside the Makefile. You can enable multi-GPU training simply by setting the `CUDA_VISIBLE_DEVICES` to multiple devices. The framework will automatically switch to the parallel training mode. For example, `export CUDA_VISIBLE_DEVICES='0,1,2'`, will use 3 devices for training, and the `batch_size` is divided by 3 for each device.

## Citation

```
@inproceedings{hsiao2024confronting,
    title={Confronting Ambiguity in 6D Object Pose Estimation via Score-Based Diffusion on SE(3)},
    author={Hsiao, Tsu-Ching and Chen, Hao-Wei and Yang, Hsuan-Kung and Lee, Chun-Yi},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={352--362},
    year={2024}
}
```
