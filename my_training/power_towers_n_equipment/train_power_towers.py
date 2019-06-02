import numpy as np

import cv2
import os
import io
import sys
from keras_retinanet.bin.train import main as trainRetina


def train(trainAnnotationsFile, classMappingFile, valAnnotationFile=None, batchSize=2,
          snapshot=None,
          weights=None,
          randomTransform=False):
    args = ['--workers=0',
            f'--batch-size={batchSize}',
            '--epochs=50',
            '--steps=1000',
            '--snapshot-path=/mnt/HDD/keras-retinanet-power_towers-snapshots']
    if snapshot:
        args.append(f'--snapshot={snapshot}')
    elif weights is not None:
        args.append(f'--weights={weights}')
    else:
        args.append('--weights=../../snapshots/ResNet-50-model.keras.h5')
    if randomTransform:
        args.append('--random-transform')
    args.extend(['csv',
                 trainAnnotationsFile,
                 classMappingFile
                 ])
    if valAnnotationFile is not None:
        args.append(f'--val-annotations={valAnnotationFile}')
    return trainRetina(args)


def main():
    trainAnnotations, valAnnotations, classMapping = './data/train.csv', './data/val.csv', './data/class_mapping.csv'

    ret = train(trainAnnotations, classMapping, valAnnotations, batchSize=2,
                snapshot=None, # '/mnt/HDD/keras-retinanet-power_towers-snapshots/1/resnet50_csv_42.h5'
                weights=None,  # '../../snapshots/resnet50_coco_best_v2.1.0.h5',
                randomTransform=False)

if __name__ == '__main__':
    main()
