from pickled_annotations import PickledAnnotations
import cv2
import numpy as np


def framePos(video):
    return int(video.get(cv2.CAP_PROP_POS_FRAMES))


def drawAnnotations(frame, annotations, color):
    for _, x1, y1, x2, y2, label, score in annotations:
        x1, y1, x2, y2 = np.round([x1, y1, x2, y2], 0).astype(np.int32)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def main():
    videoAnnotations = PickledAnnotations.video2()
    video = cv2.VideoCapture('d:/DiskE/Computer_Vision_Task/Video_2.mp4')
    while True:
        pos = framePos(video)
        ret, frame = video.read()
        if not ret:
            break
        if pos in videoAnnotations:
            drawAnnotations(frame, videoAnnotations[pos], (0, 0, 255))
        cv2.putText(frame, str(pos), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200))
        cv2.imshow('video', frame)
        if cv2.waitKey() in (27, -1):
            break
    video.release()


if __name__ == '__main__':
    main()
