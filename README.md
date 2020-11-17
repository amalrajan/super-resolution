# Super Resolution 

TensorFlow and OpenCV implementation of ESRGAN (enhanced SRGAN) [1] for upscaling video files


## What's so new? 💭

Existing ESRGAN implementations for upscaling **video** files are based on PyTorch. There exists a couple of them which uses Tensorflow with FFMPEG directly.

This repository is built on the top of **Tensorflow and OpenCV**. Useful when you'd want to integrate this on the top of a project built using the above mentioned tech without additional dependencies.

## Demo 📺

![comparison](utils/comparison.gif)

`144p` video on the left vs upscaled `576p` on the right

## Model

You can either train your own, or [use one from TensorFlow hub](https://tfhub.dev/captain-pool/esrgan-tf2/1).

## References

[1] [Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

## License
`The MIT License`
