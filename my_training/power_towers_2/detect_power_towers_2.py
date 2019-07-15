import keras
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
# import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    return tf.Session(config=config)


def main():
    keras.backend.tensorflow_backend.set_session(get_session())
    model_path = '/mnt/HDD/training_checkpoints/keras_retinanet/power_towers_2/inference_11.h5'

    # Resnet depth, must be one of 18, 34, 50, 101, 152
    model = models.load_model(model_path, backbone_name='resnet50')

    videoSource = cv2.VideoCapture('/mnt/HDD/10_кВ_Нахабино_CUT.avi')

    # videoTarget = videoWriter(videoSource, '/mnt/HDD/10_кВ_Нахабино_CUT_detections.avi')
    videoTarget = None

    while True:
        ret, frame = videoSource.read()
        if not ret:
            break
        image = predict_on_image(model, frame, thresh=0.5)
        image = putFramePos(image, videoSource.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.imshow('Video', resize(image, .5))
        if videoTarget: videoTarget.write(image)
        if cv2.waitKey() in (-1, 27):
            break
    videoSource.release()
    if videoTarget: videoTarget.release()


class LabelColor:
    blue = (255, 0, 0)
    red = (0, 0, 255)
    ankTowerLable = 0
    intmTowerLable = 1
    __labelColors = {ankTowerLable: blue, intmTowerLable: red}

    def __call__(self, label):
        return self.__labelColors[label]


LabelColor = LabelColor()


def predict_on_image(model, image, thresh):
    # copy to draw on
    draw = image.copy()

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < thresh:
            break

        color = LabelColor(label)
        b = np.round(box, 0).astype(int)
        draw_box(draw, b, color=color, thickness=1)

        # caption = f"{labels_to_names[label]} {score:.2f}"
        # draw_caption(draw, b, caption)
        if score < 1.0:
            draw_caption(draw, b, str(int(score * 100)), fontScale=0.7)

    return draw


def draw_caption(image, box, caption, fontScale=1):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """

    cv2.putText(image, caption, (box[0], box[1] + 7), cv2.FONT_HERSHEY_PLAIN, fontScale, (0, 0, 0), 2)
    cv2.putText(image, caption, (box[0], box[1] + 7), cv2.FONT_HERSHEY_PLAIN, fontScale, (255, 255, 255), 1)


def putFramePos(frame, pos):
    cv2.putText(frame, str(int(pos)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    return frame


def resize(img, factor):
    dsize = tuple(np.multiply(img.shape[1::-1], factor).astype(int))
    return cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)


def videoWriter(videoCapture: cv2.VideoCapture, videoPath):
    cc = cv2.VideoWriter_fourcc(*'MP4V')  # 'XVID' ('M', 'J', 'P', 'G')
    # videoOut = cv2.VideoWriter('/mnt/HDD/Rec_15_720_out_76.mp4', fourcc, videoIn.fps(), videoIn.resolution())
    w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    return cv2.VideoWriter(videoPath, cc, videoCapture.get(cv2.CAP_PROP_FPS), size)


if __name__ == '__main__':
    main()
