
# python generate_tfrecord.py -x ./data/train -l ./data/label_map.pbtxt -o ./data/train.record -c ./data/train/train.csv
""" Sample TensorFlow txt-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x txt_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x txt_DIR, --txt_dir txt_DIR
                        Path to the folder where the input .txt files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as txt_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
"""

import os
import glob
import pandas as pd
import io
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow txt-to-TFRecord converter")
parser.add_argument("-x",
                    "--txt_dir",
                    help="Path to the folder where the input .txt files are stored.",
                    type=str)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as txt_DIR.",
                    type=str, default=None)
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.txt_dir

label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)
inv_label_map_dict = {v:k for k,v in label_map_dict.items()}

def txt_to_csv(path):
    """Iterates through all .txt files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .txt files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    txt_list = []
    for txt_file in glob.glob(path + '/*.txt'):
        jpg_file = txt_file.replace("txt", "jpg")
        im = Image.open(jpg_file)
        width, height = im.size
        lines = open(txt_file, 'r').readlines()
        for line in lines:
            values = line.strip().split(' ')
            label_class, xmin, ymin, xmax, ymax = values
            txt_list.append([jpg_file, width, height, int(label_class)+1,
                             float(xmin), float(ymin), float(xmax), float(ymax) ])
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    txt_df = pd.DataFrame(txt_list, columns=column_name)
    return txt_df


def class_int_to_text(row_label):
    return inv_label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        ########### ADDITIONAL CHECKS START HERE ###################

        xmn = float(row['xmin'] / width)
        if xmn < 0.0:
            xmn = 0.0
        elif xmn > 1.0:
            xmn = 1.0

        xmx = float(row['xmax'] / width)
        if xmx < 0.0:
            xmx = 0.0
        elif xmx > 1.0:
            xmx = 1.0
        xmn_new = min([xmn, xmx])
        xmx_new = max([xmn, xmx])

        ymn = float(row['ymin'] / height)
        if ymn < 0.0:
            ymn = 0.0
        elif ymn > 1.0:
            ymn = 1.0

        ymx = float(row['ymax'] / height)
        if ymx < 0.0:
            ymx = 0.0
        elif ymx > 1.0:
            ymx = 1.0
        ymn_new = min([ymn, ymx])
        ymx_new = max([ymn, ymx])

        xmins.append(xmn_new)
        xmaxs.append(xmx_new)
        ymins.append(ymn_new)
        ymaxs.append(ymx_new)
        classes_text.append(class_int_to_text(row['class']).encode('utf8'))
        classes.append(row['class'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():

    writer = tf.io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = txt_to_csv(args.txt_dir)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(args.csv_path))


if __name__ == '__main__':
    main()