#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import csv
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from pandas import DataFrame
import os
import sys
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# device='cuda:0'
import torch.utils.data as Data
import time

DAYS_FOR_TRAIN = 128
label = 'final_label'
model = 'cnn'


def data(name):
    #test
    print("enter data() in cnn_712_pytorch.py")

    criteria = ["warning_line", "warning_method", "warning_abstract_method"]
    codes = []
    for criterion in criteria:
        #test
        print("criterion: ", criterion)
        path = "./data/" + name + ".xlsx"
        df = pd.read_excel(io=path)
        for i in range(0, len(df)):
            code = df.loc[i, criterion]
            if not isinstance(code, str):
                continue
            code = code.lower()
            stopwords = ['{', '}', "'", '"', "=", '(', ')', ";", ",", '\n', ':', '\\', '!', '?']
            code = list(code)
            for j in range(0, len(code)):
                if code[j] in stopwords:
                    code[j] = ' '
            code = ''.join(code)
            codes.append(code)

        path = "./data/" + name + "_" + criterion + ".txt"
        with open(path, "w") as f:
            f.writelines(codes)

        model = Word2Vec(
            sentences=LineSentence(open(path, 'r', encoding='utf8')),
            sg=0,
            vector_size=128,
            window=5,
            min_count=1
        )

        dic = model.wv.index_to_key

        df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)

        import re
        all = []
        path = "./data/" + name + ".xlsx"
        df = pd.read_excel(io=path)

        for i in range(0, len(df)):
            code = df.loc[i, criterion]
            if not isinstance(code, str):
                continue
            code = code.lower()
            if df.loc[i, label] == "TP":
                label_value = 1
            elif df.loc[i, label] == "FP":
                label_value = 0
            code = re.split('\\\\|:|\"|{|}|\'|=|\(|\)|,|;|\n|\?|!| ', code)
            while "" in code:
                code.remove("")
            vec_list = []
            for word in code:
                if word == "_minevictabl":
                    word = "_minevictableidletimemillis"
                if word in model.wv: 
                    # 跳过不在模型词汇表中的单词，避免 KeyError 错误
                    vec_list.append(model.wv[word])
            if len(vec_list) == 0:
                continue
            else:
                lists = sum(vec_list) / len(vec_list)
                lists = lists.tolist()
            warning_num = "warning" + str(i)
            lists.append(warning_num)
            lists.append(label_value)
            all.append(lists)

        result_path = "./data/" + name + "_" + criterion + ".csv"
        DataFrame(all).reset_index(drop=True).to_csv(result_path, index=None)
    print("exit data() in cnn_712_pytorch.py")


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.fc1 = nn.Linear(64, 2)
#         self.dropout = nn.Dropout(0.15)

#     def forward(self, x):
#         x = F.tanh(self.conv1(x))
#         x = F.max_pool2d(x, kernel_size=1)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.dropout(x)
#         return F.softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 使用1D卷积层，输入通道数为1，输出通道数为64，卷积核大小为3
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 全连接层，将卷积后的数据映射到2个类别
        # self.fc1 = nn.Linear(64, 2)
        self.fc1 = nn.Linear(64 * 128, 2)
        
        # Dropout层
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # 将输入的二维数据扩展为三维，以符合Conv1d的要求 (batch_size, channels, n_features)
        x = x.unsqueeze(1)  # 增加一个维度： (batch_size, 1, 128)
        
        # 通过卷积层 (Conv1d)
        x = F.tanh(self.conv1(x))  # 经过Tanh激活
        
        # 扁平化处理：把特征展平为一维向量
        x = torch.flatten(x, 1)
        
        # 通过全连接层
        x = self.fc1(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 输出经过Softmax处理的概率分布
        return F.softmax(x, dim=1)

def create_dataset(data) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集

        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。

        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    data = pd.DataFrame(data=data[0:, 0:])
    dataset_x = data.iloc[:, 0:128]
    dataset_y = data.iloc[:, 128:129]
    return (np.array(dataset_x), np.array(dataset_y))

def run(rules, name, tool, cri):
    """
        :param rules: 作者名
        :param name: 数据集名
        :param tool: 工具名(java/c)
        :param cri: 标准名
    """
    # 加载完整训练集（原train+test合并）
    path = f'./data/{tool}_{cri}.csv'
    data_set = pd.read_csv(path).drop('128', axis=1)
    full_data = data_set.drop('129', axis=1)
    full_labels = data_set['129'].astype('float32').values

    # 加载验证集（保持不变）
    # val_path = f'./data/validation_{tool}_{cri}.csv'
    # val_data = pd.read_csv(val_path).drop(['128','129'], axis=1).values.astype('float32')
    # val_labels = pd.read_csv(val_path)['129'].astype('float32').values

    # 初始化交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_data)):
        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(full_data.iloc[train_idx])
        X_test = scaler.transform(full_data.iloc[test_idx])
        # X_val = scaler.transform(val_data)

        # 转换为Tensor
        train_data = torch.from_numpy(X_train).float()
        train_labels = torch.from_numpy(full_labels[train_idx]).float()
        test_data = torch.from_numpy(X_test)
        test_labels = torch.from_numpy(full_labels[test_idx])
        # val_tensor = torch.from_numpy(X_val)
        # val_label_tensor = torch.from_numpy(val_labels)

        # 数据加载器
        train_loader = Data.DataLoader(
            dataset=Data.TensorDataset(train_data, train_labels),
            batch_size=16,
            shuffle=True,
            num_workers=2
        )

        # 模型训练（保持原训练逻辑）
        model = CNN()
        optimizer = optim.SGD(model.parameters(), lr=0.005)
        loss_function = nn.CrossEntropyLoss()
        
        # 训练循环
        model.train()
        for epoch in range(300):
            for seq, labels in train_loader:
                optimizer.zero_grad()
                y_pred = model(seq).squeeze()
                if name == "digester":
                    y_pred = y_pred.squeeze(-1)
                labels = labels.long()
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()
            

        # 验证过程（保持原验证逻辑）
        model.eval()
        with torch.no_grad():
            # val_pred = model(val_tensor)
            # val_loss = loss_function(val_pred, val_label_tensor)
            test_data = test_data.float()
            test_pred = model(test_data)
            test_labels = test_labels.long()
            test_loss = loss_function(test_pred, test_labels)
            # print(f"Fold {fold+1} Val Loss: {val_loss.item():.4f}")
            print(f"Fold {fold+1} Test Loss: {test_loss.item():.4f}")

        # 测试集评估
        with torch.no_grad():
            y_pred = torch.round(model(test_data)).numpy()
            fold_results.append({
                'true': full_labels[test_idx],
                'pred': y_pred
            })

    # 汇总所有fold结果
    all_true = np.concatenate([r['true'] for r in fold_results])
    all_pred = np.concatenate([r['pred'] for r in fold_results])
    
    # 选择第二列作为正类概率（softmax 输出的第二列表示正类的概率）
    all_pred = all_pred[:, 1]
    # 将概率转换为标签，概率大于 0.5 则为正类（1），否则为负类（0）
    all_pred = (all_pred > 0.5).astype(int)
    # 计算最终指标（保持原输出逻辑）
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred)
    recall = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    auc = roc_auc_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)

    # 保持原结果写入格式
    with open(r'./result0.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data1 = [name,tool, rules, "cnn", tp, fp, tn, fn, accuracy, 
                precision, recall, f1, auc, mcc, cri]
        wf.writerow(data1)



if __name__ == '__main__':
    # data_close = pd.read_csv('../data/token/word2vec_data/all_warning_line.csv').drop('32', axis=1)  # 读取文件

    # dataset_x, dataset_y = create_dataset(data_close)
    # print(dataset_x)
    # print("____________")
    #
    # zscore = preprocessing.StandardScaler()
    # # 标准化处理
    # dataset_x = zscore.fit_transform(dataset_x)
    # print(dataset_x)
    #
    # # 划分训练集和测试集，70%作为训练集
    # train_size = int(len(dataset_x) * 0.8)
    #
    # train_x = dataset_x[:train_size]
    # train_y = dataset_y[:train_size].flatten()
    # test_x = dataset_x[train_size:]
    # test_y = dataset_y[train_size:].flatten()
    #
    # # 转为pytorch的tensor对象
    # train_x = torch.from_numpy(train_x)
    # train_y = torch.from_numpy(train_y)
    # print(train_x)
    # print(train_y)
    # names = ['configuration','all', 'bcel', 'codec', 'collections', 'dbcp', 'digester', 'fileupload', 'net', 'pool',
    #          'mavendp']
    # import multiprocessing as mp
    # mp.set_start_method('spawn')

    # 保存原始stdout
    original_stdout = sys.stdout

    tools = ['cppcheck', 'infer', 'csa', 'spotbugs']
    tools = ['csa', 'spotbugs']
    for tool in tools:
        # data(name)
        # 打开一个文件，准备写入
        with open(f'output_{tool}_{model}.txt', 'w') as f:
            # 将stdout重定向到文件
            sys.stdout = f

            criteria = ["warning_line", "warning_method", "warning_abstract_method"]
            if tool == 'csa':
                criteria = ["warning_method", "warning_abstract_method"]
            for criterion in criteria:
                run("MengyaoZhang", '712', tool, criterion)
        # 恢复stdout为原始值
        sys.stdout = original_stdout