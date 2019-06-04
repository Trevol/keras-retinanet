import csv
from itertools import groupby
import pickle


def framePosFn(row):
    return row[0]


def toTypedRow(csvRow):
    framePos, x1, y1, x2, y2, label, score = csvRow
    return int(framePos), float(x1), float(y1), float(x2), float(y2), int(label), float(score)


def getDetectionsDict(csvPath):
    with open(csvPath) as file:
        reader = csv.reader(file, delimiter=',')
        reader = map(toTypedRow, reader)
        reader = sorted(reader, key=framePosFn)
        reader = ((framePos, list(rows)) for framePos, rows in groupby(reader, key=framePosFn))
        framesDetections = dict(reader)
        return framesDetections


class PickledAnnotations:
    @staticmethod
    def __load(pclFile):
        with open(pclFile, mode='rb') as f:
            return pickle.load(f)

    @classmethod
    def video6(cls):
        return cls.__load('data/detections_video6.pcl')

    @classmethod
    def video2(cls):
        return cls.__load('data/detections_video2.pcl')


def main():
    frameDetections = getDetectionsDict('data/detections_video2.csv')
    with open('data/detections_video2.pcl', mode='wb') as f:
        pickle.dump(frameDetections, f, pickle.HIGHEST_PROTOCOL)

    frameDetections = getDetectionsDict('data/detections_video6.csv')
    with open('data/detections_video6.pcl', mode='wb') as f:
        pickle.dump(frameDetections, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
