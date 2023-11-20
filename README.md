# Super Resolution 

- Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)

- For earlier version, please check [srgan release](https://github.com/tensorlayer/srgan/releases) and [tensorlayer](https://github.com/tensorlayer/TensorLayer).

- For more computer vision applications, check [TLXCV](https://github.com/tensorlayer/TLXCV)


### ESRGAN Architecture


![image](https://github.com/NITHISH74/Mini-project-Super-Resolution/assets/94164665/e15f91d3-2704-4093-bfd2-c15089d2a8a2)
![image](https://github.com/NITHISH74/Mini-project-Super-Resolution/assets/94164665/b6bab2aa-5857-4f8d-ab56-22a60a610657)


### Prepare Data and Pre-trained:

- 1. You need to download the pretrained model weights in [here](https://drive.google.com/file/d/1CLw6Cn3yNI1N15HyX99_Zy9QnDcgP3q7/view?usp=sharing).
- 2. You need to have the high resolution images for training.
  -  In this experiment, I used images from [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/), so the hyper-paremeters in `config.py` (like number of epochs) are seleted basic on that dataset, if you change a larger dataset you can reduce the number of epochs. 
  -  If you dont want to use DIV2K dataset, you can also use [Yahoo MirFlickr25k](http://press.liacs.nl/mirflickr/mirdownload.html), just simply download it using `train_hr_imgs = tl.files.load_flickr25k_dataset(tag=None)` in `main.py`. 
  -  If you want to use your own images, you can set the path to your image folder via `config.TRAIN.hr_img_path` in `config.py`.



### Run

You need install [TensorLayerX](https://github.com/tensorlayer/TensorLayerX#installation) at first!

Please install TensorLayerX via source

```bash
pip install git+https://github.com/tensorlayer/tensorlayerx.git 
```

#### Train
- Set your image folder in `config.py`, if you download [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/) dataset, you don't need to change it. 
- Other links for DIV2K, in case you can't find it : [test\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X4.zip), [train_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [train\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip), [valid_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_valid_HR.zip), [valid\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip).

```python
config.TRAIN.img_path = "your_image_folder/"
```
Your directory structure should look like this:

```
esrgan/
    └── config.py
    └── srgan.py
    └── train.py
    └── vgg.py
    └── model
          └── vgg19.npy
    └── DIV2K
          └── DIV2K_train_HR
          ├── DIV2K_train_LR_bicubic
          ├── DIV2K_valid_HR
          └── DIV2K_valid_LR_bicubic

```

- Start training.

```bash
python train.py
```

Modify a line of code in **train.py**, easily switch to any framework!

```python
import os
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'pytorch'
```
 We will support PyTorch as Backend soon.




```
esrgan/
    └── config.py
    └── srgan.py
    └── train.py
    └── vgg.py
    └── model
          └── vgg19.npy
    └── DIV2K
          ├── DIV2K_train_HR
          ├── DIV2K_train_LR_bicubic
          ├── DIV2K_valid_HR
          └── DIV2K_valid_LR_bicubic
    └── models
          ├── g.npz  # You should rename the weigths file. 
          └── d.npz  # If you set os.environ['TL_BACKEND'] = 'tensorflow',you should rename srgan-g-tensorflow.npz to g.npz .

```

- Start evaluation.
```bash
python train.py --mode=eval
```

Results will be saved under the folder srgan/samples/. 

### Results

![ESRGAN_DIV2K](https://github.com/NITHISH74/Mini-project-Super-Resolution/assets/94164665/159a190b-14f8-4f72-bd37-8622aad873b1)



### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Is the deconvolution layer the same as a convolutional layer ?](https://arxiv.org/abs/1609.07009)



### Citation
If you find this project useful, we would be grateful if you cite the TensorLayer paper：

```
@article{tensorlayer2017,
author = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
journal = {ACM Multimedia},
title = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
url = {http://tensorlayer.org},
year = {2017}
}

@inproceedings{tensorlayer2021,
  title={TensorLayer 3.0: A Deep Learning Library Compatible With Multiple Backends},
  author={Lai, Cheng and Han, Jiarong and Dong, Hao},
  booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={1--3},
  year={2021},
  organization={IEEE}
}
```

### Other Projects

- [Style Transfer](https://github.com/tensorlayer/adaptive-style-transfer)
- [Pose Estimation](https://github.com/tensorlayer/openpose)
### Conclusion
In conclusion, The Deep learning-based image enhancement process
using the Enhanced Super Resolution Generative Adversarial Network
(ESRGAN). The script begins by preprocessing and enhancing an
original image through the ESRGAN model, resulting in a High Resolution, visually improved image. Post-processing steps involve
sharpening the enhanced image to further refine its quality.
Additionally, the script explores down sampling and subsequently
upscaling a test image, demonstrating the model's performance on
different resolutions.


The conclusion of the script involves a visual comparison of the original
image, downscaled image, and the ESRGAN super resolution output. It
calculates the Peak Signal to Noise Ratio (PSNR) to quantitatively
measure the quality improvement achieved by the ESRGAN model.
Finally, the sharpened and enhanced output image is displayed,
providing a visual representation of the overall image enhancement
process. The script emphasizes the effectiveness of ESRGAN in Super Resolution tasks and highlights the improvements in image quality
achieved through deep learning techniques.

