import numpy as np
import scipy.io as scio
from random import shuffle

def read_data(folder, subject_ID):

    sdn = 'data_' + subject_ID
    sdp = folder + sdn + '.mat'
    sln = 'label_' + subject_ID
    slp = folder + sln + '.mat'

    # 读取 源域、目标域 数据集和标签
    all_data = scio.loadmat(sdp)[sdn]
    all_label = scio.loadmat(slp)[sln]

    # 计算样本量，训练集、测试集数目
    n_sample = np.size(all_data, 0)
    datasets_list = list(range(0, n_sample))
    shuffle(datasets_list)  # 改变后的数据集编号
    tr_num = round((n_sample/10) * 7)  # 训练集数据大小
    va_num = round((n_sample/10 * 2))  # 验证集数据大小
    te_num = round((n_sample/10) * 1)  # 验证集数据大小

    # 构建训练集、验证集、测试集 数据和标签
    train_data = all_data[datasets_list[0:tr_num], :, :]
    _train_label = all_label[datasets_list[0:tr_num], :]
    validation_data = all_data[datasets_list[tr_num:tr_num + va_num], :, :]
    _validation_label = all_label[datasets_list[tr_num:tr_num + va_num], :]
    test_data = all_data[datasets_list[tr_num + va_num:n_sample], :, :]
    _test_label = all_label[datasets_list[tr_num + va_num:n_sample], :]

    train_label = np.array(range(0, len(_train_label)))
    for i in range(0, len(_train_label)):
        train_label[i] = _train_label[i]
    validation_label = np.array(range(0, len(_validation_label)))
    for i in range(0, len(_validation_label)):
        validation_label[i] = _validation_label[i]
    test_label = np.array(range(0, len(_test_label)))
    for i in range(0, len(_test_label)):
        test_label[i] = _test_label[i]

    train_label = train_label - 1
    validation_label = validation_label - 1
    test_label = test_label - 1

    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    validation_data = np.transpose(np.expand_dims(validation_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    return train_data, train_label, validation_data, validation_label, test_data, test_label, va_num, te_num
