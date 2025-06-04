#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import csv
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from pandas import DataFrame
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch import nn
# device='cuda:0'
import torch.utils.data as Data
import time

DAYS_FOR_TRAIN = 128

def data(name):
    criteria = ["warning_line", "warning_method", "warning_abstract_method"]
    codes = []
    # 先提取所有的文本，丢入word2vec模型
    for criterion in criteria:
        path = "./data/" + name + ".xlsx"
        df = pd.read_excel(io=path)
        for i in range(0, len(df)):
            code = df.loc[i, criterion].lower()
            if code == None:
                continue
            stopwords = ['{', '}', "'", '"', "=", '(', ')', ";", ",", '\n', ':', '\\', '!', '?']
            code = list(code)
            for j in range(0, len(code)):
                if code[j] in stopwords:
                    code[j] = ' '
            code = ''.join(code)
            codes.append(code)

        path = "./data/"+name+"_"+criterion+".txt"
        # print(codes)
        f=open(path,"w")
        f.writelines(codes)
        f.close()

        model = Word2Vec(
            sentences=LineSentence(open(path, 'r', encoding='utf8')),
            sg=0,
            vector_size=128,
            window=5,
            min_count=1
        )

        dic = model.wv.index_to_key
        # print(dic)

        # model.wv.save_word2vec_format('data.vector', binary=False)
        # vector = gensim.models.KeyedVectors.load_word2vec_format('data.vector')
        # print(vector)
        # Get the vocabulary and corresponding vectors

        df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)

        import re
        all = []
        k = 0
        # 先对齐warning和label
        path = "./data/"+name+".xlsx"
        df = pd.read_excel(io=path)

        for i in range(0, len(df)):
            code = df.loc[i, criterion].lower()
            if code == None:
                continue
            if df.loc[i, 'final_label'] == "TP":
                label = 1
            elif df.loc[i, 'final_label'] == "FP":
                label = 0
            code = re.split('\\\\|:|\"|{|}|\'|=|\(|\)|,|;|\n|\?|!| ', code)
            while "" in code:
                code.remove("")
            vec_list = []
            for word in code:
                if word == "_minevictabl":
                    word = "_minevictableidletimemillis"
                vec_list.append(model.wv[word])
            if len(vec_list) == 0:
                continue
            else:
                lists = sum(vec_list) / len(vec_list)
                lists = lists.tolist()
            warning_num = "warning" + str(i)
            lists.append(warning_num)
            lists.append(label)
            all.append(lists)

        result_path = "./data/"+name+"_"+criterion+".csv"
        DataFrame(all).reset_index(drop=True).to_csv(result_path, index=None)


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
    """
    data = pd.DataFrame(data=data[0:, 0:])
    dataset_x = data.iloc[:, 0:128]
    dataset_y = data.iloc[:, 128:129]
    return (np.array(dataset_x), np.array(dataset_y))

def run(rules, name, cri):
    path = './data/train'+'_'+cri+'.csv'
    data_set = pd.read_csv(path).drop('128', axis=1)  # 读取文件
    train_data = data_set.drop('129', axis=1).reset_index(drop=True)
    train_labels = data_set['129'].reset_index(drop=True).astype('float32').values

    path = './data/test'+'_'+cri+'.csv'
    data_set = pd.read_csv(path).drop('128', axis=1)  # 读取文件
    test_data = data_set.drop('129', axis=1).reset_index(drop=True)
    test_labels = data_set['129'].reset_index(drop=True).astype('float32').values

    path = './data/validation'+'_'+cri+'.csv'
    data_set = pd.read_csv(path).drop('128', axis=1)  # 读取文件
    validation_data = data_set.drop('129', axis=1).reset_index(drop=True)
    validation_labels = data_set['129'].reset_index(drop=True).astype('float32').values

    scaler = StandardScaler(copy=False)
    train_data = pd.DataFrame(scaler.fit_transform(train_data)).astype('float32').values
    test_data = pd.DataFrame(scaler.fit_transform(test_data)).astype('float32').values
    validation_data = pd.DataFrame(scaler.fit_transform(validation_data)).astype('float32').values

    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels.flatten())
    test_data = torch.from_numpy(test_data)
    test_labels = torch.from_numpy(test_labels.flatten())
    validation_data = torch.from_numpy(validation_data)
    validation_labels = torch.from_numpy(validation_labels.flatten())

    # train_data, train_labels, test_data, test_labels, validation_data, validation_labels = train_data.to(device), train_labels.to(device), test_data.to(
    #     device), test_labels.to(device), validation_data.to(device), validation_labels.to(device)

    t0 = time.time()
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_data, train_labels),
        batch_size=20,
        shuffle=True,
        num_workers=2,
    )

    model = LSTM()
    loss_function = nn.BCELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model = model.to(device)
    # loss_function = torch.nn.L1Loss().to(device)

    epochs = 400
    model.train()
    for i in range(epochs):
        train_loss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()
            if name == "digester":
                y_pred = model(seq).squeeze(-1)
            single_loss = loss_function(y_pred, labels)
            train_loss.append(single_loss)
            single_loss.backward()
            optimizer.step()
            # print("Train Step:", i, " loss: ", single_loss)
        print("Epoch:", i, " loss: ", sum(train_loss) / len(train_loss))


    t1 = time.time()
    T = t1 - t0
    print('The training time took %.2f' % (T / 60) + ' mins.')

    tt0 = time.asctime(time.localtime(t0))
    tt1 = time.asctime(time.localtime(t1))
    print('The starting time was ', tt0)
    print('The finishing time was ', tt1)

    # 开始验证
    validation_loader = Data.DataLoader(
        dataset=Data.TensorDataset(validation_data, validation_labels),
        batch_size=20,
        shuffle=True,
        num_workers=2,
    )

    model = model.eval()
    for seq, labels in validation_loader:
        y_pred = model(seq).squeeze()
        single_loss = loss_function(y_pred, labels)
        print("EVAL Step:", i, " loss: ", single_loss)

    y_pred = model(test_data)
    y_pred = torch.round(y_pred)
    y_pred = y_pred.detach().numpy()

    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1score = f1_score(test_labels, y_pred)
    accuracy = accuracy_score(test_labels, y_pred)
    auc = roc_auc_score(test_labels, y_pred)
    mcc = matthews_corrcoef(test_labels, y_pred)

    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    print(
        f'tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1score}, auc: {auc}')


    with open(r'./data/result0.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data1 = [name, rules, "lstm", tp, fp, tn, fn, accuracy, precision, recall, f1score, auc, mcc, cri]
        wf.writerow(data1)


if __name__ == '__main__':

    names = ['train', 'test', 'validation']
    for name in names:
        data(name)


    criteria = ["warning_method", "warning_abstract_method"]
    for criterion in criteria:
        run("gxt", '712', criterion)