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
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class DetectionsCSVWriter:
    def __init__(self, csvPath):
        self.file = open(csvPath, mode='w')
        self.rowsCount = 0

    def write(self, framePos, detections):
        for detection in detections:
            box, label, score = detection
            x1, y1, x2, y2 = box
            row = f'{framePos},{x1},{y1},{x2},{y2},{label},{score}'
            if self.rowsCount > 1:
                self.file.write('\n')
            self.file.write(row)
            self.rowsCount += 1

    def close(self):
        self.file.flush()
        self.file.close()


def main():
    keras.backend.tensorflow_backend.set_session(get_session())
    model_path = './snapshots/inference_2_28.h5'
    model = models.load_model(model_path, backbone_name='resnet50')

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'pin', 1: 'solder'}

    files = [
        ('/mnt/HDD/DiskE/Computer_Vision_Task/Video_6.mp4',
         '/mnt/HDD/DiskE/Computer_Vision_Task/Video_6_pins_keras_retinanet_detections.avi',
         './data/detections_video6.csv'),

        ('/mnt/HDD/DiskE/Computer_Vision_Task/Video_2.mp4',
         '/mnt/HDD/DiskE/Computer_Vision_Task/Video_2_pins_keras_retinanet_detections.avi',
         './data/detections_video2.csv')
    ]

    for sourceVideoFile, targetVideoFile, detectionsCsvFile in files:
        videoSource = cv2.VideoCapture(sourceVideoFile)
        videoTarget = videoWriter(videoSource, targetVideoFile)
        csvWriter = DetectionsCSVWriter(detectionsCsvFile)

        framePos = 0
        while True:
            ret, frame = videoSource.read()
            if not ret:
                break
            image, detections = predict_on_image(model, labels_to_names, frame, thresh=0.5)
            csvWriter.write(framePos, detections)
            image = putFramePos(image, framePos)
            cv2.imshow('Video', image)
            videoTarget.write(image)
            if cv2.waitKey(1) == 27:
                break
            framePos += 1

        videoSource.release()
        videoTarget.release()
        csvWriter.close()


def predict_on_image(model, labels_to_names, image, thresh):
    draw = image.copy()

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale
    detections = []
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < thresh:
            break
        detections.append((box, label, score))
        # color = label_color(label)
        if score < 0.99:
            color = (0, 0, 255)
        elif label == 1:  # solder
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)
        b = np.round(box, 0).astype(int)
        draw_box(draw, b, color=color, thickness=1)

        # caption = f"{labels_to_names[label]} {score:.2f}"
        # draw_caption(draw, b, caption)
        if score < 1.0:
            draw_caption(draw, b, str(int(score * 100)), fontScale=0.7)

    return draw, detections


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
