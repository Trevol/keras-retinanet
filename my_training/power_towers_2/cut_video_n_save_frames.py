import cv2
import numpy as np
import os


def framePos(video):
    return int(video.get(cv2.CAP_PROP_POS_FRAMES))


def videoWriter(videoCapture: cv2.VideoCapture, videoPath):
    cc = cv2.VideoWriter_fourcc(*'XVID')  # MP4V 'XVID' ('M', 'J', 'P', 'G')
    w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    return cv2.VideoWriter(videoPath, cc, videoCapture.get(cv2.CAP_PROP_FPS), size)


def main():
    # 30120 frames. From 10280 to 19545
    baseDir = 'D:/DiskE/PowerTowers/10_kV'
    srcVideoPath = baseDir + '/10_кВ_Нахабино.MOV'
    framesDir = baseDir + '/NAHABINO_FRAMES_NEGATIVE/'

    video = cv2.VideoCapture(srcVideoPath)

    fromFrame, toFrame = 19600, 40000
    video.set(cv2.CAP_PROP_POS_FRAMES, fromFrame)

    while True:
        pos = framePos(video)
        ret, frame = video.read()
        if not ret:
            break

        if pos >= toFrame:
            break
        f = cv2.putText(frame.copy(), str(pos), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200))
        cv2.imshow('video', f)
        key = cv2.waitKey()
        if key in (27, -1):
             break
        if key in [ord('s'), ord('S')]:
            framePath = framesDir + f'{pos:06}.jpg'
            cv2.imwrite(framePath, frame)


    video.release()
    # resultVideo.release()


if __name__ == '__main__':
    main()
