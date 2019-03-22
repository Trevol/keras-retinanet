import numpy as np
import skimage.draw
import cv2
import os
import io
import sys


def randomShapes():
    return skimage.draw.random_shapes((768, 1024), max_shapes=40, min_size=30, max_size=60,
                                      intensity_range=((100, 255),))


def generateDataset(dir, filesNum):
    classNames = set()
    annotationsFileName = os.path.join(dir, 'annotations.csv')
    with open(annotationsFileName, 'w') as annotationsFile:
        # TODO: are we needed empty file (without objects)
        for i in range(filesNum):
            img, annotations = randomShapes()
            imageFile = f'{i + 1}.jpg'
            if len(annotations) == 0:
                annotationsFile.write(f'{imageFile},,,,,\n')
            for className, ((y1, y2), (x1, x2)) in annotations:
                annotationsFile.write(f'{imageFile},{x1},{y1},{x2},{y2},{className}\n')
                classNames.add(className)
            cv2.imwrite(os.path.join(dir, imageFile), img)
    return classNames, annotationsFile


def saveClassMapping(dir, classNames):
    classNames = sorted(classNames)
    classMappingFileName = os.path.join(dir, 'class_mapping.csv')
    with open(classMappingFileName, 'w') as classMappingFile:
        for id, className in enumerate(classNames):
            classMappingFile.write(f'{className},{id}\n')
    return classMappingFileName


def generateDatasets():
    classNames1, trainAnnotations = generateDataset('dataset/train', 200)
    classNames2, valAnnotations = generateDataset('dataset/val', 50)
    classMapping = saveClassMapping('dataset', classNames1.union(classNames2))
    return trainAnnotations, valAnnotations, classMapping


def train(trainAnnotationsFile, classMappingFile, valAnnotationFile=None):
    from keras_retinanet.bin.train import main as trainRetina
    args = ['--workers=0',
            '--weights=../../snapshots/ResNet-50-model.keras.h5',
            'csv',
            trainAnnotationsFile,
            classMappingFile]
    if valAnnotationFile is not None:
        args.append(f'--val-annotations={valAnnotationFile}')
    return trainRetina(args)


def main():
    # trainAnnotations, valAnnotations, classMapping = generateDatasets()
    trainAnnotations, valAnnotations, classMapping = 'dataset/train/annotations.csv', 'dataset/val/annotations.csv', 'dataset/class_mapping.csv'

    ret = train(trainAnnotations, classMapping, valAnnotations)
    # TODO: test ellipse detection (we train only on circles)


if __name__ == '__main__':
    main()
