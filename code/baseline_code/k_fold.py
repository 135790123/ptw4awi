import csv
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from pandas import DataFrame
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as Data
import time
import cnn_712_pytorch
from sklearn.model_selection import KFold
from tqdm import tqdm  # 导入 tqdm 库

DAYS_FOR_TRAIN = 128
label = 'final_label'
unrelated = 'no'
file_name = 'c'

def run_k_fold(rules, name, cri, num_folds=5):
    path = './data/'+file_name+'.xlsx'
    data_set = pd.read_excel(path).drop(unrelated, axis=1)
    # data_set = pd.read_excel(path, sheet_name=cri).drop('no', axis=1)
    X = data_set.drop(label, axis=1).reset_index(drop=True).values
    for i in range(len(data_set)):
        if data_set.loc[i, label] == "TP" or data_set.loc[i, label] == "FP":
            if data_set.loc[i, label] == "TP":
                label_value = 1
            elif data_set.loc[i, label] == "FP":
                label_value = 0
            data_set.loc[i, label] = label_value
    y = data_set[label].reset_index(drop=True).astype('float32').values

    scaler = StandardScaler(copy=False)
    X = pd.DataFrame(scaler.fit_transform(X)).astype('float32').values

    X = torch.from_numpy(X)
    y = torch.from_numpy(y.flatten())

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{num_folds}...")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_loader = Data.DataLoader(TensorDataset(X_train, y_train), batch_size=20, shuffle=True)
        val_loader = Data.DataLoader(TensorDataset(X_val, y_val), batch_size=20, shuffle=False)

        model = cnn_712_pytorch.CNN()
        optimizer = optim.SGD(model.parameters(), lr=0.005)
        loss_function = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(300):
            train_loss = []
            i = 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/300", unit="batch") as pbar:
                for inputs, labels in train_loader:
                    print("Train Step:", i, " loss: ", single_loss)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels.long())
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update(1)
                    i += 1

            print(f"Epoch {epoch+1} - Loss: {sum(train_loss) / len(train_loss)}")

        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                y_pred.extend(predictions.numpy())
                y_true.extend(labels.numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        print(f"Fold {fold+1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1score}, AUC: {auc}")

        with open(r'./data/result0.csv', mode='a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data1 = [name, rules, "cnn", tp, fp, tn, fn, accuracy, precision, recall, f1score, auc, mcc, cri]
            wf.writerow(data1)

if __name__ == '__main__':
    cnn_712_pytorch.data(file_name)

    criteria = ["warning_method", "warning_abstract_method"]
    for criterion in criteria:
        print(f"run_k_fold with Criterion: {criterion}")
        run_k_fold("gxt", '712', criterion)
        print(f"exit run_k_fold with Criterion: {criterion}")
