<h1 align="center">Super Resolution</h1>

<p align="center">
  Video enhancement system leveraging ESRGAN and OpenCV to upscale frames and revitalize videos with superior visual quality.
</p>


## Table of Contents

- [Flow Diagram](#flow-diagram)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


## Flow Diagram

![Flow Diagram](https://ik.imagekit.io/5jrct2yttdr/super-resolution_p7zyYNAIk.png?updatedAt=1691186321944)

## What's different about this? 

Existing ESRGAN implementations for upscaling **video** files are based on PyTorch. There exists a couple of them which uses Tensorflow with FFMPEG directly.

This repository is built on **Tensorflow and OpenCV**. Useful when you'd want to integrate this on the top of a project built already using the above mentioned tech without the need for additional dependencies.

## Demo 

![comparison](https://ik.imagekit.io/5jrct2yttdr/gifs/ezgif-2-abf8bfad96_1AROSaOTL.gif?updatedAt=1691187172670)


Side by side comparison of an original `144p` video upscaled 4x using BICUBIC interpolation on left and ESRGAN on right, achieving a PSNR value of **37.10**

## Model 

You can either fine-tune your own on a base model, or [use one from TensorFlow hub](https://tfhub.dev/captain-pool/esrgan-tf2/1).

## Installation 

Use conda / pip to create a Python 3.7 virtual environment.

`pip install -r requirements.txt`

**Note** for Windows users: Use `numpy==1.19.3`

## License

[Apace 2.0](https://choosealicense.com/licenses/apache-2.0/)
