import glob
import os
import shutil

import cv2
import tensorflow as tf
import tensorflow_hub as hub
from logzero import logger
from PIL import Image


def capture_frames(source_video: str, dest_dir: str):
    """Capture frames from a video source

    :param source: path to source video file
    :type source: str
    :param dest_dir: destination directory to extract frames to
    :type dest_dir: str
    """
    video_object = cv2.VideoCapture(source_video)
    success = 1
    count = 1

    while success:
        success, image = video_object.read()
        try:
            cv2.imwrite(f'{dest_dir}/frame-{str(count).zfill(7)}.jpg', image)
        except cv2.error:
            return
        count += 1


def combine_frames(source_dir: str, dest_dir: str):
    """Combine frames from source directory to destination directory

    :param source_dir: source directory containing images
    :type source_dir: str
    :param dest_dir: destination directory to save video to
    :type dest_dir: str
    """
    fps = 30.0
    frames = []
    # files = [f for f in os.listdir(source_dir) if os.path.isfile(
    #     os.path.join(source_dir, f))]
    files = glob.glob(source_dir + '/*')
    files.sort()

    for fl in files:
        # fl = source_dir + fl
        logger.debug('Combining frame -> ' + fl)
        image = cv2.imread(fl)
        height, width, layers = image.shape
        size = (width, height)
        frames.append(image)

    out = cv2.VideoWriter(dest_dir + '/output.mp4', 0x7634706d, fps, size)

    for frame in frames:
        out.write(frame)

    out.release()


def preprocess_image(image_path: str):
    """Loads image from path and preprocesses to make it model ready

    :type image_path: str
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(
        hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)

    return tf.expand_dims(hr_image, 0)


def save_image(image: Image.Image, filename: str):
    """Saves unscaled Tensor Images

    :param image: 3D image tensor. [height, width, channels]
    :type image: Image.Image
    :param filename: file title
    :type filename: str
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    return "%s.jpg" % filename


def make_dir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        shutil.rmtree(dir)
        os.makedirs(dir)


os.environ['TFHUB_DOWNLOAD_PROGRESS'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger.debug(str(tf.config.list_physical_devices('GPU')))

SAVED_MODEL_PATH = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
model = hub.load(SAVED_MODEL_PATH)

extracted_dir = 'extracted/'
upscaled_dir = 'upscaled/'

source_video_file_path = '/mnt/d/Dev/images/sourcevideo.mp4'
dest_video_file_path = 'upscaled_video/'

make_dir(upscaled_dir)
make_dir(extracted_dir)
make_dir(dest_video_file_path)

logger.info('Capturing frames from source video: ' + source_video_file_path)
capture_frames(source_video_file_path, extracted_dir)

files = glob.glob(extracted_dir + '/*.jpg')
files.sort()
logger.debug('Current working directory: ' + str(os.getcwd()))


logger.info('Found ' + str(len(files)) + ' frames')
logger.info('Upscaling frames from: ' + extracted_dir)
for iteration, frame in enumerate(files):
    low_res = preprocess_image(frame)
    super_res = tf.squeeze(model(low_res))
    super_res_image_path = os.path.basename(frame).split('.')[0]

    output_file = save_image(super_res, upscaled_dir + '/' + super_res_image_path)
    logger.debug('Upscaled frame -> ' + str(output_file))

    iteration += 1

logger.info('Combining frames from: ' + upscaled_dir)
combine_frames(upscaled_dir, 'upscaled_video/')
