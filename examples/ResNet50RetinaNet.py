import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# cell 2
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                   79: 'toothbrush'}


def randomUint8Batch(shape):
    return np.random.random_integers(0, 255, shape).astype(np.uint8)


def blackUint8Batch(shape):
    return np.zeros(shape, np.uint8)


def realImageBatch(file, batchLen):
    return np.array([read_image_bgr(file) for _ in range(batchLen)])


def preprocessBatch(batch):
    return preprocess_image(batch)


def predict_on_batch():
    batchShape = [32, 600, 800, 3]

    batch = preprocessBatch(blackUint8Batch(batchShape))
    # warmup
    t0 = time.time()
    boxes, scores, labels = model.predict_on_batch(batch)
    print(time.time() - t0)
    print('WARMUP ENDED!')

    t0 = time.time()
    boxes, scores, labels = model.predict_on_batch(batch)
    print(time.time() - t0)

    t0 = time.time()
    boxes, scores, labels = model.predict_on_batch(batch)
    print(time.time() - t0)

    # ---------------------------------------
    batch = preprocessBatch(randomUint8Batch(batchShape))
    # warmup
    t0 = time.time()
    boxes, scores, labels = model.predict_on_batch(batch)
    print(time.time() - t0)

    t0 = time.time()
    boxes, scores, labels = model.predict_on_batch(batch)
    print(time.time() - t0)

    # ---------------------------------------
    batch = preprocessBatch(realImageBatch('000000008021.jpg', batchShape[0]))
    # warmup
    t0 = time.time()
    boxes, scores, labels = model.predict_on_batch(batch)
    print(time.time() - t0)

    t0 = time.time()
    boxes, scores, labels = model.predict_on_batch(batch)
    print(time.time() - t0)

    t0 = time.time()
    boxes, scores, labels = model.predict_on_batch(batch)
    print(time.time() - t0)

    print('--------------------------')
    for batchNum, (bb, ss, ll) in enumerate(zip(boxes, scores, labels)):
        for b, s, l in zip(bb, ss, ll):
            if s < 0.5:
                continue
            print(batchNum, l, labels_to_names[l], s, b)

    # ---------------------------------------
    batch = realImageBatch('000000008021.jpg', batchShape[0])
    # warmup
    boxes, scores, labels = model.predict_on_batch(batch)

    print('--------------------------')
    for batchNum, (bb, ss, ll) in enumerate(zip(boxes, scores, labels)):
        for b, s, l in zip(bb, ss, ll):
            if s < 0.5:
                continue
            print(batchNum, l, labels_to_names[l], s, b)

    return
    # warmup detector
    # image = read_image_bgr(imageFiles[0])
    # image = preprocess_image(image)
    # image, scale = resize_image(image)
    # boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    for imageFile in imageFiles:
        image = read_image_bgr(imageFile)

        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=800 // 2.5, max_side=1333 // 2.5)
        print(image.shape, image.dtype)

        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        processingTime = time.time() - start
        print(f'PROC_TIME:  {processingTime}')


def predict_on_images():
    imageFiles = ['000000008021.jpg', 'dog.jpg', 'eagle.jpg', 'giraffe.jpg', 'horses.jpg', 'kite.jpg', 'person.jpg',
                  'scream.jpg']

    image = read_image_bgr(imageFiles[0])
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    for imageFile in imageFiles:
        # load image
        image = read_image_bgr(imageFile)

        # copy to draw on
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=800 // 2.5, max_side=1333 // 2.5)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        processingTime = time.time() - start

        print(f"{imageFile} processing time: {processingTime}")
        results = [(s, l) for b, s, l in zip(boxes[0], scores[0], labels[0]) if s >= 0.5]
        results.sort(key=lambda item: item[1])
        print(f'  {results}')


def predict_on_image():
    image = read_image_bgr('000000008021.jpg')
    # image = read_image_bgr('dog.jpg')
    # image = read_image_bgr('eagle.jpg')
    # image = read_image_bgr('kite.jpg')

    # image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

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
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


# predict_on_batch()
# predict_on_images()
predict_on_image()
