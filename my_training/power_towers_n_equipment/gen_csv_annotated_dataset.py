import cv2
import os
import xml.etree.ElementTree as ET
from itertools import groupby
import random


class Video:
    def __init__(self, file):
        self.cap = cv2.VideoCapture(file)

    def saveFrame(self, framePos, framesDir):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, framePos)
        ret, frame = self.cap.read()
        if not ret:
            raise Exception(f'Can not read frame at pos {framePos}')
        cv2.imwrite(self.frameFilePath(framePos, framesDir), frame)

    def saveFrames(self, framePositions, framesDir):
        for framePos in framePositions:
            self.saveFrame(framePos, framesDir)

    @staticmethod
    def frameFilePath(framePos, framesDir):
        if framesDir[-1] != '/':
            framesDir += '/'
        return framesDir + f'{framePos:06}.png'

    def release(self):
        self.cap.release()


class AnnotationsDump:
    def __init__(self, file):
        # track->box[ outside!=1 ] <-> exclude box with outside=1
        self.root = ET.parse(file).getroot()

    def labels(self):
        labelNameNodes = self.root.findall('meta/task/labels/label/name')
        return set(n.text for n in labelNameNodes)

    @staticmethod
    def text2int(t):
        return int(round(float(t)))

    def annotatedFrames(self):
        def streamBoxes():
            for tr in self.root.findall('track'):
                for b in tr.findall('box'):
                    if b.get('outside') != '0':
                        continue
                    yield int(b.get('frame')), \
                          self.text2int(b.get('xtl')), self.text2int(b.get('ytl')), \
                          self.text2int(b.get('xbr')), self.text2int(b.get('ybr')), \
                          tr.get('label')

        def frame(b):
            return b[0]

        for framePos, boxesIter in groupby(sorted(streamBoxes()), key=frame):
            yield framePos, list(boxesIter)


def trainValSplit(items, trainRatio):
    random.shuffle(items)
    trainLen = int(round(len(items) * trainRatio))
    return items[:trainLen], items[trainLen:]


def saveCsv(items, framesDir, csvFileName):
    with open(csvFileName, mode='w') as f:
        for framePos, annotatedBoxes in items:
            imagePath = Video.frameFilePath(framePos, framesDir)
            if len(annotatedBoxes) == 0:
                f.write(f'{imagePath},,,,,\n')
                continue
            for _, xtl, ytl, xbr, ybr, label in annotatedBoxes:
                l = f'{imagePath},{xtl},{ytl},{xbr},{ybr},{label}\n'
                f.write(l)


def saveClassMapping(labels, csvFilePath):
    with open(csvFilePath, mode='w') as f:
        for i, label in enumerate(labels):
            l = f'{label},{i}\n'
            f.write(l)


def prepareNegativeSamples(annotatedFramePositions, trainRatio):
    unannotatedFrames = [(framePos, []) for framePos in range(30, max(annotatedFramePositions) + 1) if
                         framePos not in annotatedFramePositions]
    random.shuffle(unannotatedFrames)
    unannotatedFrames = unannotatedFrames[: len(unannotatedFrames) // 4]
    return [framePos for framePos, _ in unannotatedFrames], trainValSplit(unannotatedFrames, trainRatio)


def main():
    framesDir = 'data/frames720'
    annotations720 = AnnotationsDump('data/PowerTowers_720.xml')

    labelSet = annotations720.labels()

    annotatedFrames = list(annotations720.annotatedFrames())
    annotatedFramePositions = set(pos for pos, _ in annotations720.annotatedFrames())
    unannotatedFramePositions, (unannotatedTrainItems, unannotatedValItems) = prepareNegativeSamples(
        annotatedFramePositions)

    video = Video('D:/DiskE/PowerTowers/Rec_15_720.mp4')
    # video.saveFrames(annotatedFramePositions, framesDir)
    video.saveFrames(unannotatedFramePositions, framesDir)

    trainItems, valItems = trainValSplit(annotatedFrames, .9)

    trainItems.extend(unannotatedTrainItems)
    valItems.extend(unannotatedValItems)

    saveCsv(trainItems, framesDir, 'data/train.csv')
    saveCsv(valItems, framesDir, 'data/val.csv')
    saveClassMapping(sorted(labelSet), 'data/class_mapping.csv')

    # TODO: save some frames without annotations (between 0 and maxAnnotatedFrame) and distribute between


def main_no_val():
    framesDir = 'data/frames720'
    annotations720 = AnnotationsDump('data/PowerTowers_720_2.xml')

    labelSet = annotations720.labels()

    annotatedFrames = list(annotations720.annotatedFrames())
    annotatedFramePositions = set(pos for pos, _ in annotations720.annotatedFrames())
    unannotatedFramePositions, (unannotatedTrainItems, unannotatedValItems) = prepareNegativeSamples(
        annotatedFramePositions, 1)

    videoPath = '/mnt/HDD/Rec_15_720.mp4'  # videoPath = 'D:/DiskE/PowerTowers/Rec_15_720.mp4'
    video = Video(videoPath)
    video.saveFrames(annotatedFramePositions, framesDir)
    video.saveFrames(unannotatedFramePositions, framesDir)

    trainItems, valItems = trainValSplit(annotatedFrames, 1)

    trainItems.extend(unannotatedTrainItems)
    valItems.extend(unannotatedValItems)

    saveCsv(trainItems, framesDir, 'data/train.csv')
    # saveCsv(valItems, framesDir, 'data/val.csv')
    saveClassMapping(sorted(labelSet), 'data/class_mapping.csv')


if __name__ == '__main__':
    main_no_val()
