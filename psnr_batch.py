import glob
import math
import os

import cv2
import numpy as np


def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)

    if mse == 0:
        return 100

    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr


os.chdir('frames2/')
lr_images = glob.glob('*.jpg')
os.chdir('SR/')
sr_images = glob.glob('*.jpg')
os.chdir('../')

summation = 0
count = len(lr_images)

for idx, img in enumerate(lr_images):
    print(f'Processing image {idx}...')
    lres = cv2.imread(img)
    lres_bicubic_interpolation = cv2.resize(
        lres, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    hres = cv2.imread(f'SR/{img}')

    val = calculate_psnr(hres, lres_bicubic_interpolation)
    summation += val

print(f'PSNR: {summation / count}')
