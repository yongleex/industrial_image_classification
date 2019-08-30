"""
AOI检测数据预处理
1. 根据标签将不同的图片移动到不同类别的文件夹中（尽管有些消耗硬盘寿命，穷人就是这么...衰）
2. 统计每个文件夹中的数量，随机复制样本，保持类别数量均衡（尽管方式有些简单，知识少的人就是这么...纯粹）
yong lee (liyong.cv@gmail.com)
2019年08月30日
"""

import os
import shutil
import random
import numpy as np


def transfer(annotation_file, image_dir, output_dir, clean_flag=False):
    """
    根据标签文件，将image目录下的图像按照类别移动到output目录下放好
    """
    if clean_flag:
        if not os.path.exists(output_dir):  # 不存在就创建一个
            os.makedirs(output_dir)
        shutil.rmtree(output_dir)  # 清除数据
        os.mkdir(output_dir)

    with open(annotation_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            path, label = line.split(",")
            src_path = os.path.join(image_dir, path)
            dst_path = os.path.join(output_dir, str(label[:-1]) + "/" + path)
            if not os.path.exists(os.path.dirname(dst_path)):
                os.mkdir(os.path.dirname(dst_path))
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(src_path + "...路径不存在")
            pass

    # 把其他无关类别的文件夹删除掉
    for dir_name in os.listdir(output_dir):
        if dir_name not in ["0", "1", "2", "3", "4", "5"]:
            shutil.rmtree(os.path.join(output_dir, dir_name))


def data_split(output_dir, train_dir, val_dir, ratio_val=0.05):
    """
    把output_dir文件夹中图像进行分派到train_dir, val_dir
    """
    shutil.rmtree(val_dir)
    shutil.rmtree(train_dir)
    os.mkdir(val_dir)
    os.mkdir(train_dir)
    dir_names = os.listdir(output_dir)
    for dir_name in dir_names:
        file_names = os.listdir(os.path.join(output_dir, dir_name))
        random.shuffle(file_names)
        move_length = np.int(ratio_val * len(file_names))
        for move_file in file_names[:move_length]:
            src_path = os.path.join(output_dir, dir_name + os.sep + move_file)
            dst_path = os.path.join(val_dir, dir_name + os.sep + move_file)
            if not os.path.exists(os.path.dirname(dst_path)):
                os.mkdir(os.path.dirname(dst_path))
            shutil.copy(src_path, dst_path)
        for move_file in file_names[move_length:]:
            src_path = os.path.join(output_dir, dir_name + os.sep + move_file)
            dst_path = os.path.join(train_dir, dir_name + os.sep + move_file)
            if not os.path.exists(os.path.dirname(dst_path)):
                os.mkdir(os.path.dirname(dst_path))
            shutil.copy(src_path, dst_path)


def train_data_balance(train_dir):
    """
    那个类别的文件夹下图片数量少，那我就多复制几次，使得类别数据达到数量平衡
    """
    # 获取每个文件夹下图像的数量
    dir_names = os.listdir(train_dir)
    dir_file_number = []
    for dir_name in dir_names:
        dir_file_number.append(len(os.listdir(os.path.join(train_dir, dir_name))))
    max_num = np.max(dir_file_number)

    # 针对每个文件夹进行复制
    for dir_name in dir_names:
        dir_name = os.path.join(train_dir, dir_name)
        file_names = os.listdir(dir_name)
        random.shuffle(file_names)
        for i in range(max_num - len(file_names)):
            j = i % len(file_names)
            src_path = os.path.join(dir_name, file_names[j])
            dst_path = os.path.join(dir_name, str(i) + "_" + file_names[j])
            shutil.copy(src_path, dst_path)


def preprocess():
    # ------------------------------------------------------------------------------------------------------------
    # 执行的功能设置
    trans_flag = False
    split_flag = True
    balance_flag = True

    # 相关路径设置
    train_label_file = "../data/train.csv"
    test_label_file = "../data/new_test.csv"
    train_image_dir = "../data/train_images"
    test_image_dir = "../data/test_images"
    output_dir = "../data/temp/all"

    train_data_dir = "../data/temp/train"
    valid_data_dir = "../data/temp/val"

    # -----------------------------------------------------------------------------------------------------------
    # 转移图像到类别文件夹
    if trans_flag:
        transfer(train_label_file, train_image_dir, output_dir, clean_flag=True)
        transfer(test_label_file, test_image_dir, output_dir, clean_flag=False)

    # 分派一些数据作为验证集
    if split_flag:
        data_split(output_dir=output_dir, train_dir=train_data_dir, val_dir=valid_data_dir, ratio_val=0.08)

    # 训练数据样本均衡 by copy
    if balance_flag:
        train_data_balance(train_dir=train_data_dir)


if __name__ == '__main__':
    preprocess()
