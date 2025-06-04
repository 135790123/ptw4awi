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
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch import nn
import torch.optim as optim
# device='cuda:0'
import torch.utils.data as Data
import time

DAYS_FOR_TRAIN = 128
label = 'final_label'
model = 'lstm'


def data(name):
    #test
    print("enter data() in lstm_712.py")

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
    print("exit data() in lstm_712.py")



class LSTM(nn.Module):
    """
        使用LSTM进行回归

        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size=128, hidden_size=100, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        hidden_cell = (torch.zeros(1, 1, self.hidden_size),  # shape: (n_layers, batch, hidden_size)
                       torch.zeros(1, 1, self.hidden_size))
        lstm_out, (h_n, h_c) = self.lstm(input_x, hidden_cell)
        linear_out = self.linear(lstm_out.view(len(input_x), -1))  # =self.linear(lstm_out[:, -1, :])
        predictions = self.sigmoid(linear_out)
        return predictions

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
    # 加载完整数据集（合并原train和test）
    path = f'./data/{tool}_{cri}.csv'
    data_set = pd.read_csv(path).drop('128', axis=1)
    full_data = data_set.drop('129', axis=1)
    full_labels = data_set['129'].astype('float32').values

    # 加载独立验证集（保持不变）
    # val_path = f'./data/validation_{tool}_{cri}.csv'
    # val_data = pd.read_csv(val_path).drop(['128','129'], axis=1).values.astype('float32')
    # val_labels = pd.read_csv(val_path)['129'].astype('float32').values

    # 初始化交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_data)):
        # 数据标准化（每折独立进行）
        scaler = StandardScaler()
        X_train = scaler.fit_transform(full_data.iloc[train_idx])
        X_test = scaler.transform(full_data.iloc[test_idx])
        # X_val = scaler.transform(val_data)

        # 转换为Tensor
        train_tensor = torch.from_numpy(X_train).float()
        train_label_tensor = torch.from_numpy(full_labels[train_idx]).float()
        test_tensor = torch.from_numpy(X_test).float()
        test_label_tensor = torch.from_numpy(full_labels[test_idx])
        # val_tensor = torch.from_numpy(X_val).float()
        # val_label_tensor = torch.from_numpy(val_labels).float()

        # 数据加载器
        train_loader = Data.DataLoader(
            dataset=Data.TensorDataset(train_tensor, train_label_tensor),
            batch_size=16,
            shuffle=True,
            num_workers=2
        )

        # 模型初始化
        model = LSTM()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.BCELoss()

        # 训练循环
        model.train()
        for epoch in range(400):
            epoch_loss = []
            for seq, labels in train_loader:
                optimizer.zero_grad()
                y_pred = model(seq).squeeze()
                if name == "digester":
                    y_pred = y_pred.squeeze(-1)
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            print(f"Fold {fold+1} Epoch {epoch}: Loss {sum(epoch_loss)/len(epoch_loss):.4f}")

        # 验证过程
        model.eval()
        with torch.no_grad():
            # 独立验证集评估
            test_pred = model(test_tensor)
            test_loss = loss_function(test_pred.squeeze(), test_label_tensor)
            print(f"Fold {fold+1} Validation Loss: {test_loss.item():.4f}")

            # 测试集预测
            test_pred = torch.round(model(test_tensor)).detach().numpy()
            fold_results.append({
                'true': full_labels[test_idx],
                'pred': test_pred.squeeze()
            })

    # 汇总所有fold结果
    all_true = np.concatenate([r['true'] for r in fold_results])
    all_pred = np.concatenate([r['pred'] for r in fold_results])

    # 计算最终指标
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred)
    recall = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    auc = roc_auc_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)

    # 结果写入（保持原格式）
    with open(r'./result0.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data1 = [name, tool, rules, "lstm", tp, fp, tn, fn, accuracy, 
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
    # names = ['train', 'test', 'validation']

    # 保存原始stdout
    original_stdout = sys.stdout

    tools = ['spotbugs', 'cppcheck', 'infer', 'csa']
    tools = ['infer']
    for tool in tools:
        # data(name)
        # 打开一个文件，准备写入
        with open(f'output_{tool}_{model}.txt', 'w') as f:
            # 将stdout重定向到文件
            sys.stdout = f

            criteria = [ "warning_method", "warning_abstract_method"]
            for criterion in criteria:
                run("MengyaoZhang", '712', tool, criterion)
        # 恢复stdout为原始值
        sys.stdout = original_stdout