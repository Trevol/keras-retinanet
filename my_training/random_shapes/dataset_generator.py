import numpy as np
import skimage.draw
import cv2
import os
import io


def randomShapes():
    return skimage.draw.random_shapes((768, 1024), max_shapes=40, min_size=30, max_size=60,
                                      intensity_range=((100, 255),))


def generateDataset(dir, filesNum):
    annotationsFileName = os.path.join(dir, 'annotations.csv')
    with open(annotationsFileName, 'w') as annotationsFile:
        for i in range(filesNum):
            img, annotations = randomShapes()
            imageFile = f'{i + 1}.jpg'
            for category, ((y1, y2), (x1, x2)) in annotations:
                annotationsFile.write(f'{imageFile},{x1},{y1},{x2},{y2},{category}')
            cv2.imwrite(imageFile, os.path.join(dir, imageFile))


def main():
    generateDataset('dataset/train', 200)


if __name__ == '__main__':
    main()
