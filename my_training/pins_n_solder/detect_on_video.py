import keras
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
# import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def main():
    keras.backend.tensorflow_backend.set_session(get_session())
    # model_path = './snapshots/1/resnet50_pins_n_solder_12_inference.h5'
    model_path = './snapshots/resnet50_pins_n_solder_14_inference.h5'
    model = models.load_model(model_path, backbone_name='resnet50')

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'pin', 1: 'solder', 2: 'anomaly_bent_solder'}

    video = cv2.VideoCapture('/media/trevol/HDD/DiskE/Computer_Vision_Task/Video_6.mp4')
    video.set(cv2.CAP_PROP_POS_FRAMES, 269)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        image = predict_on_image(model, labels_to_names, frame, thresh=0.96)
        image = putFramePos(image, video.get(cv2.CAP_PROP_POS_FRAMES))
        image = resize(image, .5)
        cv2.imshow('Video', image)
        if cv2.waitKey() == 27:
            break
    video.release()


def predict_on_image(model, labels_to_names, image, thresh):
    # copy to draw on
    draw = image.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB, dst=draw)

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

        # color = label_color(label)
        if score < 0.99:
            color = (0, 0, 255)
        elif label == 1:
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)
        b = np.round(box, 0).astype(int)
        draw_box(draw, b, color=color, thickness=1)

        caption = f"{labels_to_names[label]} {score:.2f}"
        draw_caption(draw, b, caption)

    # return cv2.cvtColor(draw, cv2.COLOR_RGB2BGR, dst=draw)
    return draw


def putFramePos(frame, pos):
    cv2.putText(frame, str(int(pos)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    return frame


def resize(img, factor):
    dsize = tuple(np.multiply(img.shape[1::-1], factor).astype(int))
    return cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    main()
