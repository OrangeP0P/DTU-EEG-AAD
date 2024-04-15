import numpy as np
import scipy.io as scio
from random import shuffle

def read_target_data(folder, target_subject_ID):

    sdn = 'data_' + target_subject_ID
    sdp = folder + sdn + '.mat'
    sln = 'label_' + target_subject_ID
    slp = folder + sln + '.mat'

    # 读取 源域、目标域 数据集和标签
    all_data = scio.loadmat(sdp)[sdn]
    all_label = scio.loadmat(slp)[sln]

    # 计算样本量，训练集、测试集数目
    n_sample = np.size(all_data, 0)
    datasets_list = list(range(0, n_sample))
    shuffle(datasets_list)  # 改变后的数据集编号
    tr_num = int(n_sample / (8 + 2) * 8)  # 训练集数据大小
    va_num = int(n_sample - tr_num)  # 验证集数据大小

    # 构建训练集、验证集、测试集 数据和标签
    train_target_data = all_data[datasets_list[0:tr_num], :, :]
    _train_target_label = all_label[datasets_list[0:tr_num], :]
    validation_target_data = all_data[datasets_list[tr_num + 1:tr_num + va_num], :, :]
    _validation_target_label = all_label[datasets_list[tr_num + 1:tr_num + va_num], :]

    train_target_label = np.array(range(0, len(_train_target_label)))
    for i in range(0, len(_train_target_label)):
        train_target_label[i] = _train_target_label[i]
    validation_target_label = np.array(range(0, len(_validation_target_label)))
    for i in range(0, len(_validation_target_label)):
        validation_target_label[i] = _validation_target_label[i]

    train_target_label = train_target_label - 1
    validation_target_label = validation_target_label - 1

    train_target_data = np.transpose(np.expand_dims(train_target_data, axis=1), (0, 1, 3, 2))
    validation_target_data = np.transpose(np.expand_dims(validation_target_data, axis=1), (0, 1, 3, 2))

    return train_target_data, train_target_label, validation_target_data, validation_target_label, va_num


def read_source_data(folder, source_subject_ID):

    sdn = 'data_' + source_subject_ID
    sdp = folder + sdn + '.mat'
    sln = 'label_' + source_subject_ID
    slp = folder + sln + '.mat'

    # 读取 源域、目标域 数据集和标签
    all_data = scio.loadmat(sdp)[sdn]
    all_label = scio.loadmat(slp)[sln]

    # 计算样本量，训练集、测试集数目
    n_sample = np.size(all_data, 0)
    datasets_list = list(range(0, n_sample))
    shuffle(datasets_list)  # 改变后的数据集编号

    # 构建训练集、验证集、测试集 数据和标签
    source_data = all_data[datasets_list, :, :]
    _source_label = all_label[datasets_list, :]

    source_label = np.array(range(0, len(_source_label)))
    for i in range(0, len(_source_label)):
        source_label[i] = _source_label[i]
    source_label = source_label - 1
    source_data = np.transpose(np.expand_dims(source_data, axis=1), (0, 1, 3, 2))
    return source_data, source_label