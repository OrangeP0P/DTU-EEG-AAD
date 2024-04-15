from torch.autograd import Variable
import numpy as np
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
from Model_128Hz import choose_net
from argparse import ArgumentParser
from DataLoaderWS import read_data
from Model_128Hz import handle_param
from ProgressBar import progress_bar
import RandomSeed
import time
import csv
import pandas as pd
import math

subject_ID = '1'  # 设置测试用户编号
Decision_window = '01s'  # 修改 Decision window即可选取对应任务
net_name = 'Frame2_' + Decision_window  # 选取网络
folder = 'D:\EEG Audio Dataset\Processed Dataset\DTU Data/128Hz/Processed EEG data ' + Decision_window + ' no-filter downsampled 128Hz/'
print('训练网络名称：', net_name, '\n')

SEED = 2024  # 起始种子数，固定后不变，设置为2024年，也可以设置为自己的幸运数字
torch.set_num_threads(1)  # 限制cpu线程数，解决占用率过高问题
RandomSeed.setup_seed(SEED)  # 设置固定种子数 2024年，并生成随机参数
start = time.time()  # 记下开始时刻，统计程序运行时间

Epoch = 200  # 训练轮数，可自定义
Cross_Mission = 0  # 后期交叉验证用

Cross_Aver_ACC = []  # 交叉验证准确率

parser = ArgumentParser()
parser.add_argument("-b", "--batch", help="batch size", type=int, default=128)
parser.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=1e-3)
parser.add_argument("-ep", "--epochs", help="your training target", type=int, default=Epoch)
parser.add_argument("-opt", "--optimizer", help="adam | rmsp", type=str, default='adam')
parser.add_argument("-lf", "--loss-function", help="loss function", type=str, default='CrossEntropy')
parser.add_argument("-act", "--activation-function", help="elu | relu | lrelu", type=str, default='relu')
parser.add_argument("-m", "--model", help="eeg | dcn", type=str, default=net_name)
parser.add_argument("-load", "--load", help="your pkl file path", type=str, default='')
args = parser.parse_args()

# 读取训练模型数据
train_data, train_label, validation_data, validation_label, test_data, test_label, va_num, te_num = (
    read_data(folder, subject_ID))

source_data = Data.TensorDataset(torch.from_numpy(train_data.astype(np.float32)),
                                 torch.from_numpy(train_label.astype(np.float32)))
validation_data = Data.TensorDataset(torch.from_numpy(validation_data.astype(np.float32)),
                                     torch.from_numpy(validation_label.astype(np.float32)))
source_loader = Data.DataLoader(dataset=source_data, batch_size=args.batch, shuffle=True)
validation_loader = Data.DataLoader(dataset=validation_data, batch_size=args.batch, shuffle=True)
epoch_num = args.epochs  # 从网络参数中读取epoch的数目
net_dict = choose_net(args)  # 选取网络

# 初始化损失矩阵，用来存储损失
TOTAL_LOSS = []
CLF_LOSS = []
MMD_LOSS = []
VA_LOSS = []
VA_MMD_LOSS = []
VA_CLF_LOSS = []
ACC = []  # 储存每个Batch-size的准确率

# net[0]:model, net[1]:optimizer, net[2]:loss_function
for key, net in net_dict.items():
    optimizer, loss_func = handle_param(args, net[0])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350, 450,
                                                                            550, 650, 750, 850, 950],
                                                     gamma=0.7)
    net.extend([optimizer, loss_func, scheduler])
    total_loss = 0
    epoch_start = time.time()

net[0].train()
max_acc = 0
for epoch in range(args.epochs):
    tr_loss = 0
    progress_bar(epoch, Epoch, 0, 0)  # 程序进度条
    if (epoch+1) % 2 == 0:  # 每 5个 epoch更新一次时间计次
        epoch_end = time.time()
        epoch_start = time.time()

    iter_source = iter(source_loader)
    num_iter = len(source_loader)  # 数据长度

    net[0] = net[0].cuda()
    iter_count_source = 0
    net_count = 0

    ###################################### 开始训练网络 #########################################
    for i in range(0, num_iter):
        train_source_data_1, train_source_label_1 = next(iter_source)
        train_source_data_1, train_source_label_1 = train_source_data_1.cuda(), train_source_label_1.cuda()
        train_source_data_1, train_source_label_1 = Variable(train_source_data_1), Variable(train_source_label_1)

        for key, net in net_dict.items():
            net_count = net_count + 1
            output, mmd_loss = net[0](train_source_data_1)
            clf_loss = net[2](output, train_source_label_1.long())
            tr_loss = clf_loss + mmd_loss  # 总loss
            net[1].zero_grad()
            tr_loss.backward()  # 反传总loss
            net[1].step()
            net[3].step()

    list.append(TOTAL_LOSS, tr_loss.item())  # 将每个epoch的loss值储存进list，以绘制图片
    list.append(CLF_LOSS, clf_loss.item())

    ###################################### 开始测试网络 #########################################
    cuda = torch.cuda.is_available()
    net[0].eval()  # 开启测试模式
    va_clf_loss = 0  # 初始化测试损失
    correct = 0  # 所有分类正确的样本数
    count = 0  # 统计test中已检测样本个数

    for validation_data, validation_label in validation_loader:
        validation_data, validation_label = validation_data.cuda(), validation_label.cuda()
        validation_data, validation_label = Variable(validation_data), Variable(validation_label)
        va_pred, va_mmd_loss = net[0](validation_data)
        va_clf_loss += net[2](va_pred, validation_label.long())
        pred = va_pred.data.max(1)[1]
        _correct = pred.eq(validation_label.data.view_as(pred)).cpu().sum()  # 当前epoch中’分类正确样本数‘
        correct = correct + _correct  # 分类正确样本总数
        count = count + 1  # batch-size计数

    list.append(VA_CLF_LOSS, va_clf_loss.item())

    acc = correct / va_num  # 计算准确率
    if acc > max_acc:
        print('ACC = ', acc, '/n')
        max_acc = acc
    list.append(ACC, acc.item())

acc_max = 0  # 临时acc值，用来挑选所有epoch中最大的acc
acc_max_num = 0
loss_min_num = 0

for i in range(0, epoch_num):
    acc_current = ACC[i]
    if acc_current >= acc_max:
        acc_max = acc_current
        acc_max_num = i  # 最大acc值对应的epoch序号

###################################### 绘制ACC-Epoch，Loss-Epoch准确率 ###########################################
x_label = range(0, epoch_num)  # 画图用横坐标

plt.plot(x_label, ACC)  # 绘制验证集在训练时的 ACC 图像
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc / Epoch')
plt.savefig('Figure/Train/Cross_' + str(Cross_Mission+1) + '/'
            + 'ACC/' + subject_ID + ' ACC')
plt.close()

# 绘制训练集、验证集的 CLF_Loss
plt.figure()
plt.plot(x_label, CLF_LOSS)
plt.xlabel('Epoch')
plt.ylabel('Classification Loss')
plt.title('Classification Loss / Epoch')
plt.savefig('Figure/Train/Cross_' + str(Cross_Mission+1) + '/'
            + 'CLF_LOSS/' + subject_ID + ' CLF Loss')
plt.close()

# 绘制TOTAL_LOSS曲线图
plt.figure()
plt.plot(x_label, TOTAL_LOSS)
plt.xlabel('Epoch')
plt.ylabel('TOTAL LOSS')
plt.title('Total Loss / Epoch')
plt.savefig('Figure/Train/Cross_' + str(Cross_Mission+1) + '/'
            + 'TOTAL_LOSS/' + subject_ID + ' TOTAL LOSS')
plt.close()

print('  Max ACC = ', acc_max)


