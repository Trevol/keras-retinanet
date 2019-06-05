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


class VideoAnnotationsDump:
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
                label = tr.get('label')
                for b in tr.findall('box'):
                    if b.get('outside') != '0':
                        continue
                    frameNum = int(b.get('frame'))
                    xtl = self.text2int(b.get('xtl'))
                    ytl = self.text2int(b.get('ytl'))
                    xbr = self.text2int(b.get('xbr'))
                    ybr = self.text2int(b.get('ybr'))
                    yield frameNum, xtl, ytl, xbr, ybr, label

        def fnFrameNum(b):
            return b[0]

        for framePos, boxesIter in groupby(sorted(streamBoxes()), key=fnFrameNum):
            yield framePos, list(boxesIter)


class ImageSetAnnotationsDump:
    def __init__(self, annotationXmlFile):
        self.root = ET.parse(annotationXmlFile).getroot()
        self.__labels = None

    def labels(self):
        if self.__labels is None:
            labelNameNodes = self.root.findall('meta/task/labels/label/name')
            self.__labels = set(n.text for n in labelNameNodes)
        return self.__labels

    @staticmethod
    def text2int(t):
        return int(round(float(t)))

    def annotatedImages(self):
        def streamBoxes():
            for imageNode in self.root.findall('image'):
                imageName = imageNode.get('name')
                for boxNode in imageNode.findall('box'):
                    if boxNode.get('outside') == '1':
                        continue
                    xtl = self.text2int(boxNode.get('xtl'))
                    ytl = self.text2int(boxNode.get('ytl'))
                    xbr = self.text2int(boxNode.get('xbr'))
                    ybr = self.text2int(boxNode.get('ybr'))
                    label = boxNode.get('label')
                    yield imageName, xtl, ytl, xbr, ybr, label

        def fnImageName(b):
            return b[0]

        for imageName, boxesIter in groupby(sorted(streamBoxes()), key=fnImageName):
            yield imageName, list(boxesIter)


def trainValSplit(items, trainRatio):
    random.shuffle(items)
    trainLen = int(round(len(items) * trainRatio))
    return items[:trainLen], items[trainLen:]


def saveCsv(items, framesDir, csvFileName):
    with open(csvFileName, mode='w') as f:
        for imageName, annotatedBoxes in items:
            imagePath = os.path.join(framesDir, imageName)
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


def main():
    framesDir = 'data/frames'
    annotations = ImageSetAnnotationsDump('data/18_PinsSelectiveAnnotation.xml')

    labelSet = annotations.labels()

    annotatedImages = list(annotations.annotatedImages())
    trainItems = list(filter(lambda i: i[0] != 'f2_1763_117533.33_117.53.jpg', annotatedImages))
    valItems = list(filter(lambda i: i[0] == 'f2_1763_117533.33_117.53.jpg', annotatedImages))

    trainItems.append(('f2_0031_2066.67_2.07.jpg', []))
    valItems.append(('f2_0033_2200.00_2.20.jpg', []))

    saveCsv(trainItems, framesDir, 'data/train.csv')
    saveCsv(valItems, framesDir, 'data/val.csv')
    saveClassMapping(sorted(labelSet), 'data/class_mapping.csv')

    # TODO: save some frames without annotations (between 0 and maxAnnotatedFrame) and distribute between


if __name__ == '__main__':
    main()
