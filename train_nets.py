"""
AOI检测 神经网络训练
1. 神经网络训练那一套（数据，模型，训练，存储等。用神经网络就是这么...死板）
2. 注意下必要的数据增强手段（尽管方式有些简单，知识少的人就是这么...纯粹）
3. 我希望一次性就把五个网络训练好。
# Step.0 配置
# Step.1 数据
# Step.2 模型
# Step.3 训练
yong lee (liyong.cv@gmail.com)
2019年08月30日
"""

from __future__ import print_function
from __future__ import division
import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

from preprocessing import preprocess

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    """
    支持函数，训练模型
    """
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def train_net():
    # Step.0 配置
    data_dir = "../data/temp/"

    num_classes = 6  # Number of classes in the dataset
    batch_size = 20  # Batch size for training (change depending on how much memory you have)
    num_epochs = 20  # Number of epochs to train for
    input_size = 448

    pre_trained, feature_extract = True, True

    # 得到设备运行环境
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Step.1 数据 ------------------------------------------------------------------------------------------------------
    # -1.1 定义transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(1),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # -1.2 Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    # Step.2 模型 ------------------------------------------------------------------------------------------------------
    # -2.1 模型本身
    model_ft = models.resnet18(pretrained=pre_trained)  # 从模型库中调用训练好的模型
    # model_ft = models.resnet34(pretrained=pre_trained)

    num_ftrs = model_ft.fc.in_features  # 得到新的模型，更改fc层
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    for para in list(model_ft.parameters())[:-9]:  # 锁定前面层的参数
        para.requires_grad = False

    model_ft = model_ft.to(device)
    # print(model_ft) # 显示模型看看

    # -2.2 指定模型训练的参数
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # -2.3 Loss设计
    criterion = nn.CrossEntropyLoss()

    # Step.3 训练
    # 3.1 指定训练优化器
    optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.90)

    # -3.2 Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs)
    return model_ft

    # # -3.3 保存训练结果
    # image_example = torch.rand(1, 3, input_size, input_size).to(device)
    # traced_script_module = torch.jit.trace(model_ft, image_example)
    # traced_script_module.save("model/SeedNets_5.pt")
    #
    # # -3.4 测试单步时间
    # for i in range(10):
    #     start = time.time()
    #     result = model_ft(image_example)
    #     print(result)
    #     time_elapsed = time.time() - start
    #     print('single image cost:' + str(time_elapsed))


def main():
    for i in range(0, 5):
        preprocess()
        model = train_net()
        torch.save(model, "../data/models/ResNet_%2d.pt" % (i + 1))


if __name__ == '__main__':
    main()
