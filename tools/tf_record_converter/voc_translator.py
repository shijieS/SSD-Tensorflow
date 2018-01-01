import os
import sys
import random

import numpy as np
import tensorflow as tf
from io import StringIO
import csv
from PyQt5.QtCore import pyqtSignal, QThread, QObject

import xml.etree.ElementTree as ET

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class VocTranslator(QThread):
    current_status = pyqtSignal(object)

    def __init__(self, dataset_dir, output_dir, annotationDir = 'Annotations', imageDir = 'JPEGImages', imageFormat=b'JPEG', name='voc_train', shuffling=False, labelText=None, sampelsPerFile=200):
        super(VocTranslator, self).__init__()
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.annotationDir = annotationDir
        self.imageDir = imageDir
        self.imageFormat = imageFormat
        self.name = name
        self.shuffling = shuffling
        self.labelText = labelText
        self.sampelsPerFile = sampelsPerFile

        # TFRecords convertion parameters.
        self.RANDOM_SEED = 4242

        # VOC Labels
        self.LABELS_FILENAME = 'labels.txt'
        self.VOC_LABELS = {
            'none': (0, 'Background'),
            'aeroplane': (1, 'Vehicle'),
            'bicycle': (2, 'Vehicle'),
            'bird': (3, 'Animal'),
            'boat': (4, 'Vehicle'),
            'bottle': (5, 'Indoor'),
            'bus': (6, 'Vehicle'),
            'car': (7, 'Vehicle'),
            'cat': (8, 'Animal'),
            'chair': (9, 'Indoor'),
            'cow': (10, 'Animal'),
            'diningtable': (11, 'Indoor'),
            'dog': (12, 'Animal'),
            'horse': (13, 'Animal'),
            'motorbike': (14, 'Vehicle'),
            'person': (15, 'Person'),
            'pottedplant': (16, 'Indoor'),
            'sheep': (17, 'Animal'),
            'sofa': (18, 'Indoor'),
            'train': (19, 'Vehicle'),
            'tvmonitor': (20, 'Indoor'),
        }


    def __del__(self):
        self.wait()


    def _process_image(self, directory, name):
        """Process a image and annotation file.

        Args:
          filename: string, path to an image file e.g., '/path/to/example.JPG'.
          coder: instance of ImageCoder to provide TensorFlow image coding utils.
        Returns:
          image_buffer: string, JPEG encoding of RGB image.
          height: integer, image height in pixels.
          width: integer, image width in pixels.
        """
        # Read the image file.
        filename = directory + self.DIRECTORY_IMAGES + name + '.jpg'
        image_data = tf.gfile.FastGFile(filename, 'rb').read()

        # Read the XML annotation file.
        filename = os.path.join(directory, self.DIRECTORY_ANNOTATIONS, name + '.xml')
        tree = ET.parse(filename)
        root = tree.getroot()

        # Image shape.
        size = root.find('size')
        shape = [int(size.find('height').text),
                 int(size.find('width').text),
                 int(size.find('depth').text)]
        # Find annotations.
        bboxes = []
        labels = []
        labels_text = []
        difficult = []
        truncated = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(int(self.VOC_LABELS[label][0]))
            labels_text.append(label.encode('ascii'))

            if obj.find('difficult'):
                difficult.append(int(obj.find('difficult').text))
            else:
                difficult.append(0)
            if obj.find('truncated'):
                truncated.append(int(obj.find('truncated').text))
            else:
                truncated.append(0)

            bbox = obj.find('bndbox')
            bboxes.append((float(bbox.find('ymin').text) / shape[0],
                           float(bbox.find('xmin').text) / shape[1],
                           float(bbox.find('ymax').text) / shape[0],
                           float(bbox.find('xmax').text) / shape[1]
                           ))
        return image_data, shape, bboxes, labels, labels_text, difficult, truncated


    def _convert_to_example(self, image_data, labels, labels_text, bboxes, shape,
                            difficult, truncated):
        """Build an Example proto for an image example.

        Args:
          image_data: string, JPEG encoding of RGB image;
          labels: list of integers, identifier for the ground truth;
          labels_text: list of strings, human-readable labels;
          bboxes: list of bounding boxes; each box is a list of integers;
              specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
              to the same label as the image label.
          shape: 3 integers, image shapes in pixels.
        Returns:
          Example proto
        """
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for b in bboxes:
            assert len(b) == 4
            [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

        image_format = self.image_format
        example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': int64_feature(shape[0]),
                'image/width': int64_feature(shape[1]),
                'image/channels': int64_feature(shape[2]),
                'image/shape': int64_feature(shape),
                'image/object/bbox/xmin': float_feature(xmin),
                'image/object/bbox/xmax': float_feature(xmax),
                'image/object/bbox/ymin': float_feature(ymin),
                'image/object/bbox/ymax': float_feature(ymax),
                'image/object/bbox/label': int64_feature(labels),
                'image/object/bbox/label_text': bytes_feature(labels_text),
                'image/object/bbox/difficult': int64_feature(difficult),
                'image/object/bbox/truncated': int64_feature(truncated),
                'image/format': bytes_feature(image_format),
                'image/encoded': bytes_feature(image_data)}))
        return example


    def _add_to_tfrecord(self, dataset_dir, name, tfrecord_writer):
        """Loads data from image and annotations files and add them to a TFRecord.

        Args:
          dataset_dir: Dataset directory;
          name: Image name to add to the TFRecord;
          tfrecord_writer: The TFRecord writer to use for writing.
        """
        image_data, shape, bboxes, labels, labels_text, difficult, truncated = self._process_image(dataset_dir, name)
        example = self._convert_to_example(image_data, labels, labels_text,
                                      bboxes, shape, difficult, truncated)
        tfrecord_writer.write(example.SerializeToString())


    def _get_output_filename(self, output_dir, name, idx):
        return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


    def _get_label_from_text(self, labelText):
        if not labelText:
            return None
        else:
            reader = csv.reader(StringIO(labelText), delimiter=',')
            label = {}
            for row in reader:
                row = [i.strip() for i in row]
                key = row[0]
                index = int(row[1])
                explanation = row[2]
                label[key] = (index, explanation)
                return label

    def tranlate(self, dataset_dir, output_dir, annotationDir = 'Annotations', imageDir = 'JPEGImages', imageFormat=b'JPEG', name='voc_train', shuffling=False, labelText=None, samplesPerFile=200):
        """Runs the conversion operation.

        Args:
          dataset_dir: The dataset directory where the dataset is stored.
          output_dir: Output directory.
        """

        # Original dataset organisation.
        self.DIRECTORY_ANNOTATIONS = annotationDir+'/'
        self.DIRECTORY_IMAGES = imageDir+'/'
        if not labelText:
            self.VOC_LABELS = self._get_label_from_text(labelText)

        # Image Format
        self.image_format = imageFormat

        # Number of items packed
        self.SAMPLES_PER_FILES = samplesPerFile

        if not tf.gfile.Exists(dataset_dir):
            tf.gfile.MakeDirs(dataset_dir)

        # Dataset filenames, and shuffling.
        path = os.path.join(dataset_dir, self.DIRECTORY_ANNOTATIONS)
        filenames = sorted(os.listdir(path))
        if shuffling:
            random.seed(self.RANDOM_SEED)
            random.shuffle(filenames)

        # Process dataset files.
        i = 0
        fidx = 0
        while i < len(filenames):
            # Open new TFRecord file.
            tf_filename = self._get_output_filename(output_dir, name, fidx)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                j = 0
                while i < len(filenames) and j < self.SAMPLES_PER_FILES:
                    info = '\r>> Converting image %d/%d' % (i+1, len(filenames))
                    sys.stdout.write(info)
                    self.current_status.emit(info)
                    sys.stdout.flush()

                    filename = filenames[i]
                    img_name = filename[:-4]
                    self._add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                    i += 1
                    j += 1
                fidx += 1

        # Finally, write the labels file:
        # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
        # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
        info = '\nFinished converting the Pascal VOC dataset!'
        print(info)
        self.current_status.emit(info)

    def run(self):
        self.tranlate(self.dataset_dir, self.output_dir, self.annotationDir, self.imageDir, self.imageFormat, self.name, self.shuffling, self.labelText, self.sampelsPerFile)