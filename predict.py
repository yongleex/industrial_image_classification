"""
AOI检测 神经网络预测
1. 一张张图像读入
2. 几个网络一起预测下，投个票，直接给出结果
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import cv2
import os
import copy
import torch.nn.functional as F
from collections import Counter

from PIL import Image

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

num_classes = 6  # Number of classes in the dataset
batch_size = 16  # Batch size for training (change depending on how much memory you have)
num_epochs = 20  # Number of epochs to train for
input_size = 448

data_dir = "../data/test_images"
new_result_path = "../data/new_raw_result.csv"

# 得到设备运行环境
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step.1 数据 ----------------------------------------------------------------------------------------------------------
# -1.1 定义transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()  # ,
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Step.2 模型 ----------------------------------------------------------------------------------------------------------
models = [""] * 5
for i in range(5):
    models[i] = torch.load("../data/models/ResNet_%2d.pt" % (i + 1))
    models[i].to(device)
    models[i].eval()

# Step.3 读图预测
with open(new_result_path, "w") as f:
    f.writelines("ID,c1,c2,c3,c4,c5,ct")  # 每个网络的预测结果都保存起来

    image_names = os.listdir(data_dir)
    image_names.sort()
    for image_name in image_names:
        img = Image.open(os.path.join(data_dir, image_name))
        img = img.convert('RGB')

        img = data_transforms["val"](img)
        img = img.to(device)
        img = img.unsqueeze(0)

        result = []
        f.writelines(image_name)  # 每个网络的预测结果都保存起来
        probs = torch.tensor(0.0)
        for model in models:
            output = model(img)
            prob = F.softmax(output, dim=1)  # prob是10个分类的概率
            probs = probs + prob
            # print(prob)
            value, predicted = torch.max(output.data, 1)
            # print(predicted.item())
            print(image_name, predicted.item())
            result.append(predicted.item())
            f.writelines("," + str(predicted.item()))

        value, predicted = torch.max(probs.data, 1)
        # print(predicted.item())
        print(image_name, predicted.item())
        result.append(predicted.item())
        f.writelines("," + str(predicted.item()))

        f.writelines("\n")

# 更多的结果分析处理，见result_analysis.ipynb
# 当然路径需要改下
