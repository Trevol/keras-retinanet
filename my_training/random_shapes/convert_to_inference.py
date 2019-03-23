import os
import sys
from keras_retinanet.bin.convert_model import main as convertModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    args = [
        './snapshots/resnet50_csv_02.h5',  # model_in
        'resnet50_random_shapes_02.h5'  # model_out
    ]
    convertModel(args)


if __name__ == '__main__':
    main()
