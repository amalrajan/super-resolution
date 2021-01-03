# Super Resolution 

TensorFlow and OpenCV implementation of ESRGAN (enhanced SRGAN) [1] for upscaling video files


## 💭 What's so new? 

Existing ESRGAN implementations for upscaling **video** files are based on PyTorch. There exists a couple of them which uses Tensorflow with FFMPEG directly.

This repository is built on **Tensorflow and OpenCV**. Useful when you'd want to integrate this on the top of a project built already using the above mentioned tech without the need for additional dependencies.

## 📺 Demo 

![comparison](utils/comparison.gif)

Side by side comparison of an original `144p` video upscaled 4x using BICUBIC interpolation on left and ESRGAN on right, achieving a PSNR value of **37.10**

## 📁 Model 

You can either train your own, or [use one from TensorFlow hub](https://tfhub.dev/captain-pool/esrgan-tf2/1).

## 💿 Installation 

Use conda / pip to create a Python 3.7 virtual environment.

`pip install -r requirements.txt`

**Note** for Windows users: Use `numpy==1.19.3`

## 📚 References 

[1] [Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

## License
`Apache-2.0 License`
