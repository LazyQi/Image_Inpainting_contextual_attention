# 将原数据集分为training ,validation
import os
import random
import argparse
import numpy as np
import cv2

# 划分验证集训练集
_NUM_TEST = 1547

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='E:\CODES\generative_inpainting-master\Dataset\imagenet', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data/imagenet/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data/imagenet/validation_static_view.flist', type=str,
                    help='The validation filename.')


def _get_filenames(dataset_dir):
    # photo_filenames = []
    image_list = os.listdir(dataset_dir)
    photo_filenames = [os.path.join(dataset_dir, _) for _ in image_list]
    return photo_filenames

def load_mnist():
    # step1: 索引mnist路径
    data_dir = os.path.join('./Dataset', 'mnist')
    # step2: 分别打开训练、测试图片/标签
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)  # 读取进来的是一个一维向量
    train_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)  # 'train-images-idx3-ubyte'这个文件前十六位保存的是一些说明

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)  # 读取进来的是一个一维向量
    train_label = loaded[8:].reshape((60000)).astype(np.float)  # 'train-images-idx3-ubyte'这个文件前十六位保存的是一些说明

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_label = loaded[8:].reshape((10000)).astype(np.float)

    train_X = np.asarray(train_X)
    train_label = np.asarray(train_label)

    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_label, test_label), axis=0).astype(np.int)

    seed = 666
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    # 手动将数据转化成one-hot编码形式
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, int(y[i])] = 1.0

    return X, y_vec

if __name__ == "__main__":

    args = parser.parse_args()

    data_dir = args.folder_path

    # get all file names
    photo_filenames = _get_filenames(data_dir)
    print("size of imagenet is %d" % (len(photo_filenames)))

    # 切分数据为测试训练集
    random.seed(0)
    random.shuffle(photo_filenames)
    training_file_names = photo_filenames[_NUM_TEST:]
    validation_file_names = photo_filenames[:_NUM_TEST]

    print("training file size:", len(training_file_names))
    print("validation file size:", len(validation_file_names))

    # # make output file if not existed
    # if not os.path.exists(args.train_filename):
    #     os.makedirs(os.getcwd()+args.train_filename)
    #
    # if not os.path.exists(args.validation_filename):
    #     os.makedirs(os.getcwd()+args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename)
