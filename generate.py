import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub 
import matplotlib.pyplot as plt

os.environ['TFHUB_DOWNLOAD'] = 'True'

IMAGE_PATH = 'original.png'
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))

    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,: -1]
    
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)

    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    
    image.save(f"{filename}.jpg")
    print(f"Saved as {filename}.jpg")


hr_image = preprocess_image(IMAGE_PATH)
model = hub.load(SAVED_MODEL_PATH)

sr_image = model(hr_image)
sr_image = tf.squeeze(sr_image)

save_image(tf.squeeze(sr_image), filename='sr_image')
