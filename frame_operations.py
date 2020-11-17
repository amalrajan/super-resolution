import cv2
import os


def capture_frames(source, dest):
    video_object = cv2.VideoCapture(source)
    success = 1
    count = 1

    while success:
        success, image = video_object.read()
        try:
            cv2.imwrite(f'{dest}frame-{str(count).zfill(7)}.jpg', image)
        except cv2.error:
            return
        count += 1
    

def combine_frames(source, dest):
    fps = 30.0
    frames = []
    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    # files.sort(key=lambda x: int(x[5:-4]))

    print(files)

    for fl in files:
        fl = source + fl
        image = cv2.imread(fl)
        height, width, layers = image.shape
        size = (width, height)
        frames.append(image)

    out = cv2.VideoWriter(dest + 'output.mp4', 0x7634706d, fps, size)

    for frame in frames:
        out.write(frame)

    out.release()
