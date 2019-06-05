# /mnt/HDD/training_checkpoints/keras_retinanet/power_towers_2/inference_11.h5

from keras_retinanet.bin.convert_model import main as convertToInference

src = '/mnt/HDD/training_checkpoints/keras_retinanet/power_towers_2/resnet50_csv_11.h5'
dst = '/mnt/HDD/training_checkpoints/keras_retinanet/power_towers_2/inference_11.h5'
convertToInference([src, dst])

