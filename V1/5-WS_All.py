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

Seed_ACC = []  # 统计不同种子数时模型的最终准确率
SEED = 2024  # 起始种子数，固定后不变，设置为2024年，也可以设置为自己的幸运数字
torch.set_num_threads(1)  # 限制cpu线程数，解决占用率过高问题
RandomSeed.setup_seed(SEED)  # 设置固定种子数 2024年，并生成随机参数
start = time.time()  # 记下开始时刻，统计程序运行时间
time_step = 10
Batch_size = 128
Epoch = 500  # 训练轮数，可自定义
Cross_Mission = 0  # 后期交叉验证用

subject_ID_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
data_Type_list = ['D:\EEG Audio Dataset\Processed Dataset\DTU Data/128Hz/Processed EEG data 01s no-filter downsampled 128Hz/',
                  'D:\EEG Audio Dataset\Processed Dataset\DTU Data/128Hz/Processed EEG data 025s no-filter downsampled 128Hz/',
                  'D:\EEG Audio Dataset\Processed Dataset\DTU Data/128Hz/Processed EEG data 05s no-filter downsampled 128Hz/',
                  'D:\EEG Audio Dataset\Processed Dataset\DTU Data/128Hz/Processed EEG data 075s no-filter downsampled 128Hz/',
                  'D:\EEG Audio Dataset\Processed Dataset\DTU Data/128Hz/Processed EEG data 1s no-filter downsampled 128Hz/',
                  'D:\EEG Audio Dataset\Processed Dataset\DTU Data/128Hz/Processed EEG data 2s no-filter downsampled 128Hz/']
csv_column_name_list = ['0.1s', '0.25s', '0.5s', '0.75s', '1s', '2s']  # 数据保存csv标签行
net_name_list = ['Frame2_01s', 'Frame2_025s', 'Frame2_05s', 'Frame2_075s', 'Frame2_1s', 'Frame2_2s']
num_decision_window_task = np.size(net_name_list, 0)  # 一共需要进行的 Decision Window 任务个数
num_subject = np.size(subject_ID_list, 0)  # 用户数目


acc_results_validation = pd.DataFrame() # 初始化CSV文件，写入表头
acc_results_test_last_epoch = pd.DataFrame()
acc_results_test_min_loss = pd.DataFrame()
start_time = time.time()  # 记下开始时刻，统计程序运行时间
for TASK_decision in range(0, num_decision_window_task):
    best_model_path = 'Model/best_model.pth'  # Path to save the best model
    last_model_path = 'Model/last_model_epoch.pth'  # Path to save the last epoch model
    ACC_decision_window_validation = []  # 存放每个 decision window 的18个用户的准确率
    ACC_decision_window_test_last_epoch = []  # 存放每个 decision window 的18个用户的准确率
    ACC_decision_window_test_min_loss = []  # 存放每个 decision window 的18个用户的准确率
    net_name = net_name_list[TASK_decision]
    csv_column_name = csv_column_name_list[TASK_decision]
    folder = data_Type_list[TASK_decision]
    print('===========Decision Window', csv_column_name_list[TASK_decision], '===========')
    for ID in range(0, num_subject):
        best_loss = float('inf')  # 最小loss
        best_model = None  # 存储最优模型字典
        max_acc = 0
        subject_ID = subject_ID_list[ID]  # 选取用户
        es_time_total = 0  # 预估程序总剩余时间
        parser = ArgumentParser()
        parser.add_argument("-b", "--batch", help="batch size", type=int, default=Batch_size)
        parser.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=1e-3)
        parser.add_argument("-ep", "--epochs", help="your training target", type=int, default=Epoch)
        parser.add_argument("-opt", "--optimizer", help="adam | rmsp", type=str, default='adam')
        parser.add_argument("-lf", "--loss-function", help="loss function", type=str, default='CrossEntropy')
        parser.add_argument("-act", "--activation-function", help="elu | relu | lrelu", type=str, default='relu')
        parser.add_argument("-m", "--model", help="eeg | dcn", type=str, default=net_name)
        parser.add_argument("-load", "--load", help="your pkl file path", type=str, default='')
        args = parser.parse_args()

        # 读取训练模型数据
        train_data, train_label, validation_data, validation_label, test_data, test_label, va_num, te_num = (read_data(folder, subject_ID))
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

        net[0].train()
        epoch_start = time.time()
        es_time_run = 0
        for epoch in range(args.epochs):
            tr_loss = 0
            if (epoch+1) % time_step == 0:   # 每 5 个 epoch更新一次时间计次
                epoch_end = time.time()
                duration = epoch_end - epoch_start
                es_time_run = round((duration * ((Epoch-epoch)/time_step))/3600, 3)  # 计算所需时间
                es_time_total = round((es_time_run + (duration*(num_subject-ID-1)*(Epoch/time_step)
                           + duration*(num_decision_window_task-TASK_decision-1)*num_subject*(Epoch/time_step)))/3600,2)
                epoch_start = time.time()

            progress_bar(epoch, Epoch, es_time_run, es_time_total)  # 程序进度条

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

            ###################################### 测试验证集 #########################################
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
            list.append(ACC, acc.item())

            # Check if this is the best model
            if va_clf_loss < best_loss:
                best_loss = va_clf_loss
                best_model = net[0].state_dict()  # Save the current state_dict
                torch.save(net[0].state_dict(), best_model_path)

            if acc > max_acc:
                print('ACC = ', acc, '/n')
                max_acc = acc
            list.append(ACC, acc.item())

        list.append(ACC_decision_window_validation, max_acc.item())
        torch.save(net[0].state_dict(), last_model_path.format(TASK_decision=net_name, ID=ID + 1))

        print('\nSubject', ID + 1, '验证集 ACC = ', max_acc.item())  # 打印更新准确率

        ###################################### 测试 测试集 最终Epoch #########################################
        best_model = net[0]  # 假设你的模型结构已经被初始化为net[0]
        best_model.load_state_dict(torch.load(last_model_path))
        best_model = best_model.cuda()  # 确保模型在GPU上
        best_model.eval()  # 设置为评估模式

        test_loss = 0
        correct = 0
        total = 0

        test_data = Data.TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
                                       torch.from_numpy(test_label.astype(np.float32)))
        test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch, shuffle=False)

        with torch.no_grad():  # 测试阶段不需要计算梯度
            for data, labels in test_loader:
                data, labels = data.cuda(), labels.cuda()
                outputs, mmd_loss = best_model(data)
                loss = loss_func(outputs, labels.long())
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / te_num
        print('Subject', ID + 1, '测试集（最终Epoch） ACC = ', test_acc)  # 打印更新准确率
        list.append(ACC_decision_window_test_last_epoch, test_acc)

        ###################################### 测试 测试集 最小 Loss #########################################
        best_model = net[0]
        best_model.load_state_dict(torch.load(best_model_path))
        best_model = best_model.cuda()  # 确保模型在GPU上
        best_model.eval()  # 设置为评估模式

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # 测试阶段不需要计算梯度
            for data, labels in test_loader:
                data, labels = data.cuda(), labels.cuda()
                outputs, mmd_loss = best_model(data)
                loss = loss_func(outputs, labels.long())
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / te_num
        print('Subject', ID + 1, '测试集（最小Loss） ACC = ', test_acc)  # 打印更新准确率
        list.append(ACC_decision_window_test_min_loss, test_acc)

    mean_ACC_one_decision_window_validation = np.mean(ACC_decision_window_validation)  # 一个decision window 18个用户的平均准确率
    list.append(ACC_decision_window_validation, mean_ACC_one_decision_window_validation)  # 将平均准确率纳入 ACC_decision_window
    acc_results_validation[csv_column_name_list[TASK_decision]] = ACC_decision_window_validation

    mean_ACC_one_decision_window_test_last_epoch = np.mean(ACC_decision_window_test_last_epoch)  # 一个decision window 18个用户的平均准确率
    list.append(ACC_decision_window_test_last_epoch, mean_ACC_one_decision_window_test_last_epoch)  # 将平均准确率纳入 ACC_decision_window
    acc_results_test_last_epoch[csv_column_name_list[TASK_decision]] = ACC_decision_window_test_last_epoch

    mean_ACC_one_decision_window_test_min_epoch = np.mean(ACC_decision_window_test_min_loss)  # 一个decision window 18个用户的平均准确率
    list.append(ACC_decision_window_test_min_loss, mean_ACC_one_decision_window_test_min_epoch)  # 将平均准确率纳入 ACC_decision_window
    acc_results_test_min_loss[csv_column_name_list[TASK_decision]] = ACC_decision_window_test_min_loss

acc_results_validation.to_csv('CSV/5-ACC-DTU-Validation.csv', index=False)
acc_results_test_last_epoch.to_csv('CSV/5-ACC-DTU-Test_last_epoch.csv', index=False)
acc_results_test_min_loss.to_csv('CSV/5-ACC-DTU-Test_min_loss.csv', index=False)
end_time = time.time()  # 记下结束时刻，统计程序运行时间
total_time = round((end_time - start_time)/3600, 2)
print('Total time: ',  total_time)  # 打印更新准确率
