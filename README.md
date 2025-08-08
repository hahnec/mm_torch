# mm_torch: Mueller matrix library for PyTorch

[![arXiv paper link](https://img.shields.io/badge/paper-arXiv:2411.07918-red)](https://arxiv.org/pdf/2411.07918)

## Description

This repository provides Mueller Matrix computations for PyTorch featuring the Lu-Chipman decomposition. A reference implementation can be found in the [polar_segment repo](https://www.github.com/hahnec/polar_segment). Specifically, the [infer.py](https://github.com/hahnec/polar_segment/blob/master/infer.py) file shows how a mueller matrix model is initialized and the [train.py](https://github.com/hahnec/polar_segment/blob/master/train.py) file contains a more elaborate usage for plotting image results.

## Exemplary plots showing azimuth and brain fiber tracts

| ![Azimuth angle plot](docs/media_images_azimuth_test_4623_ca31ac53ae00aa8c88ac.png) | ![Fiber tract plot](docs/media_images_img_fiber_test_4623_5397685a7719cd03ce39.png) |
|:--------------------------:|:--------------------------:|
| **Azimuth angle map** | **Fiber tract map** |

<br>
<p align="center" style="background-color: white;">
  <img src="docs/color_bar.svg" alt="Colorbar" width="33%" />
</p>

## Citation

<pre>
@misc{hahne:2024:polar_augment,
      title={Physically Consistent Image Augmentation for Deep Learning in Mueller Matrix Polarimetry}, 
      author={Christopher Hahne and Omar Rodriguez-Nunez and Éléa Gros and Théotim Lucas and Ekkehard Hewer and Tatiana Novikova and Theoni Maragkou and Philippe Schucht and Richard McKinley},
      year={2024},
      eprint={2411.07918},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.07918}, 
} 
</pre>