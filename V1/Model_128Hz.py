import numpy as np
import torch
import torch.nn as nn
from MMD import mmd_rbf
import torch.utils.data
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):  # 深度可分离卷积
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=(1, kernel_size), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nout, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Frame2_01s(nn.Module):
    def __init__(self):
        super(Frame2_01s, self).__init__()
        self.Kernel = 4
        self.F1 = 8
        self.DF = 2
        self.Channel = 66
        self.Class = 2
        self.mapsize = 32

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3',
                                  depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 2))

    def forward(self, source_data):
        mmd_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        feature = feature.view(-1, self.mapsize)
        class_output = self.class_classifier(feature)
        return class_output, mmd_loss


class Frame2_025s(nn.Module):
    def __init__(self):
        super(Frame2_025s, self).__init__()
        self.Kernel = 10
        self.F1 = 8
        self.DF = 4
        self.Channel = 66
        self.Class = 2
        self.mapsize = 256

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 128))
        self.class_classifier.add_module('fc-2', nn.Linear(128, 64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, 2))

    def forward(self, source_data):
        mmd_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        feature = feature.view(-1, self.mapsize)
        class_output = self.class_classifier(feature)
        return class_output, mmd_loss


class Frame2_05s(nn.Module):
    def __init__(self):
        super(Frame2_05s, self).__init__()
        self.Kernel = 40
        self.F1 = 16
        self.DF = 8
        self.Channel = 66
        self.Class = 2
        self.mapsize = 1024

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.MaxPool2d(kernel_size=(1, 4)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 512))
        self.class_classifier.add_module('fc-3', nn.Linear(512, 256))
        self.class_classifier.add_module('fc-4', nn.Linear(256, 64))
        self.class_classifier.add_module('fc-5', nn.Linear(64, 2))

    def forward(self, source_data):
        mmd_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        # print(feature.shape)
        _, s2, _, s4 = feature.shape
        feature = feature.view(-1, s2*s4)
        class_output = self.class_classifier(feature)
        return class_output, mmd_loss

class Frame2_075s(nn.Module):
    def __init__(self):
        super(Frame2_075s, self).__init__()
        self.Kernel = 48
        self.F1 = 16
        self.DF = 12
        self.Channel = 66
        self.Class = 2
        self.mapsize = 2304

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 4)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 1024))
        self.class_classifier.add_module('fc-2', nn.Linear(1024, 512))
        self.class_classifier.add_module('fc-3', nn.Linear(512, 256))
        self.class_classifier.add_module('fc-4', nn.Linear(256, 64))
        self.class_classifier.add_module('fc-5', nn.Linear(64, 2))

    def forward(self, source_data):
        mmd_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        # print(feature.shape)
        _, s2, _, s4 = feature.shape
        feature = feature.view(-1, s2*s4)
        class_output = self.class_classifier(feature)
        return class_output, mmd_loss

class Frame2_1s(nn.Module):
    def __init__(self):
        super(Frame2_1s, self).__init__()
        self.Kernel = 40
        self.F1 = 16
        self.DF = 10
        self.Channel = 66
        self.Class = 2
        self.mapsize = 16*160

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 4)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 1024))
        self.class_classifier.add_module('fc-2', nn.Linear(1024, 512))
        self.class_classifier.add_module('fc-3', nn.Linear(512, 256))
        self.class_classifier.add_module('fc-4', nn.Linear(256, 64))
        self.class_classifier.add_module('fc-5', nn.Linear(64, 2))

    def forward(self, source_data):
        mmd_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        # print(feature.shape)
        _, s2, _, s4 = feature.shape
        feature = feature.view(-1, s2*s4)
        class_output = self.class_classifier(feature)
        return class_output, mmd_loss

class Frame2_2s(nn.Module):
    def __init__(self):
        super(Frame2_2s, self).__init__()
        self.Kernel = 80
        self.F1 = 16
        self.DF = 12
        self.Channel = 66
        self.Class = 2
        self.mapsize = 192 * 8

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 1024))
        self.class_classifier.add_module('fc-2', nn.Linear(1024, 512))
        self.class_classifier.add_module('fc-3', nn.Linear(512, 256))
        self.class_classifier.add_module('fc-4', nn.Linear(256, 64))
        self.class_classifier.add_module('fc-5', nn.Linear(64, 2))

    def forward(self, source_data):
        mmd_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        # print(feature.shape)
        _, s2, _, s4 = feature.shape
        feature = feature.view(-1, s2*s4)
        class_output = self.class_classifier(feature)
        return class_output, mmd_loss

class Frame2_5s(nn.Module):
    def __init__(self):
        super(Frame2_5s, self).__init__()
        self.Kernel = 10
        self.F1 = 8
        self.DF = 2
        self.Channel = 66
        self.Class = 2
        self.mapsize = 320

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.1))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.1))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 64))
        # self.class_classifier.add_module('fc-2', nn.Linear(256, 64))
        # self.class_classifier.add_module('fc-3', nn.Linear(128, 64))
        self.class_classifier.add_module('fc-4', nn.Linear(64, 2))

    def forward(self, source_data):
        mmd_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        _, s2, _, s4 = feature.shape
        feature = feature.view(-1, s2*s4)
        class_output = self.class_classifier(feature)
        return class_output, mmd_loss

'''最大均值差异计算函数'''
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

'''选取激活函数类型'''
def choose_act_func(act_name):
    if act_name == 'elu':
        return nn.ELU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    else:
        raise TypeError('activation_function type not defined.')

'''处理预定义网络和训练参数'''
def handle_param(args, net):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'rmsp':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    else:
        raise TypeError('optimizer type not defined.')
    if args.loss_function == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    else:
        raise TypeError('loss_function type not defined.')
    return optimizer, loss_function

'''选取网络和激活函数'''
def choose_net(args):
    if args.model == 'Frame2_01s':
        return {
        'elu': [Frame2_01s()]
        }
    elif args.model == 'Frame2_025s':
        return {
        'elu': [Frame2_025s()]
        }
    elif args.model == 'Frame2_05s':
        return {
        'elu': [Frame2_05s()]
        }
    elif args.model == 'Frame2_075s':
        return {
        'elu': [Frame2_075s()]
        }
    elif args.model == 'Frame2_1s':
        return {
        'elu': [Frame2_1s()]
        }
    elif args.model == 'Frame2_2s':
        return {
        'elu': [Frame2_2s()]
        }
    elif args.model == 'Frame2_5s':
        return {
        'elu': [Frame2_5s()]
        }
    else:
        raise TypeError('model type not defined.')