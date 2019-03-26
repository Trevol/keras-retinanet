"""
Converting LabelImg xml annotations to keras_retina csv annotations
"""
import os
import pathlib
import xml.etree.ElementTree as ET


def _isNoneOrEmpty(a):
    return a is None or len(a) == 0


def _in(item, a):
    if _isNoneOrEmpty(a):
        return True
    return item in a


def _notIn(item, a):
    if _isNoneOrEmpty(a):
        return True
    return item not in a


def convert(labelImgDir, outputFileName, imageExtensions=['jpg'], includeClasses=[], excludeClasses=[]):
    xmlFiles = set()
    imageFiles = []
    # ['jpg'] to ['.JPG']
    imageExtensions = [(ext if ext.startswith('.') else '.' + ext).upper() for ext in imageExtensions]

    for root, dirs, files in os.walk(labelImgDir):
        for filename in files:
            baseName, ext = os.path.splitext(filename)
            ext = ext.upper()
            if ext == '.XML':
                xmlFiles.add(baseName)
            elif ext in imageExtensions:
                imageFiles.append((filename, baseName, ext))

    annotationsFileName = os.path.join(labelImgDir, outputFileName)
    with open(annotationsFileName, mode='w') as annotationsFile:
        for imageFile, baseName, ext in imageFiles:
            imageHasAnnotations = baseName in xmlFiles
            if imageHasAnnotations:
                for className, ((x1, y1), (x2, y2)) in readAnnotations(os.path.join(labelImgDir, baseName + '.xml')):
                    if _in(className, includeClasses) and _notIn(className, excludeClasses):
                        annotationsFile.write(f'{imageFile},{x1},{y1},{x2},{y2},{className}\n')
            else:
                # write empty annotation
                annotationsFile.write(f'{imageFile},,,,,\n')


def readAnnotations(xmlAnnotations):
    e = ET.parse(xmlAnnotations).getroot()
    for object in e.findall('object'):
        className = object.find('name').text
        bbox = object.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        yield className, ((x1, y1), (x2, y2))


if __name__ == '__main__':
    def main():
        # TODO: automatically distribute files to train and val
        # prepare train
        # prepare val
        # TODO: prepare class mappings automatically
        # train

        labelImgDir = 'D:\DiskE\Computer_Vision_Task\\frames_annotations\\annotated'
        outputFile = '_keras_retina_annotations.csv'
        convert(labelImgDir, outputFile, includeClasses=['pin'])


    main()
