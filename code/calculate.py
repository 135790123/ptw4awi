import pandas as pd
import os
from sklearn.metrics import roc_auc_score, matthews_corrcoef

def load_and_concatenate_csv(file_list):
    # Load all CSV files and concatenate them into a single DataFrame
    df_list = [pd.read_csv(file) for file in file_list]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def calculate_confusion_matrix(df, true_label_col, pred_label_col):
    # Calculate TP, FP, TN, FN
    TP = ((df[true_label_col] == 1) & (df[pred_label_col] == True)).sum()
    FP = ((df[true_label_col] == 0) & (df[pred_label_col] == True)).sum()
    TN = ((df[true_label_col] == 0) & (df[pred_label_col] == False)).sum()
    FN = ((df[true_label_col] == 1) & (df[pred_label_col] == False)).sum()
    return TP, FP, TN, FN

def calculate_metrics(TP, FP, TN, FN, df, true_label_col, pred_label_col):
    # Calculate accuracy
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate AUC
    auc = roc_auc_score(df[true_label_col], df[pred_label_col])
    
    # Calculate MCC
    mcc = matthews_corrcoef(df[true_label_col], df[pred_label_col])
    
    return accuracy, precision, recall, f1, auc, mcc

def main(file_list, true_label_col='target', pred_label_col='raw_preds'):
    combined_df = load_and_concatenate_csv(file_list)
    # print(combined_df.head())
    print(f"len(combined_df): {len(combined_df)}")
    TP, FP, TN, FN = calculate_confusion_matrix(combined_df, true_label_col, pred_label_col)
    print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
    accuracy, precision, recall, f1, auc, mcc = calculate_metrics(TP, FP, TN, FN, combined_df, true_label_col, pred_label_col)
    print(f'Accuracy: {round(accuracy, 4)}')
    print(f'Precision: {round(precision, 4)}')
    print(f'Recall: {round(recall, 4)}')
    print(f'F1 Score: {round(f1, 4)}')
    print(f'AUC: {round(auc, 4)}')
    print(f'MCC: {round(mcc, 4)}')

    # print(f"{accuracy},{TP},{FP},{TN},{FN},{precision},{recall},{f1},{auc},{mcc}")
    print(f"{round(accuracy, 4)},{TP},{FP},{TN},{FN},{round(precision, 4)},{round(recall, 4)},{round(f1, 4)},{round(auc, 4)},{round(mcc, 4)}")
    print(f"{round(accuracy, 4)} {TP} {FP} {TN} {FN} {round(precision, 4)} {round(recall, 4)} {round(f1, 4)} {round(auc, 4)} {round(mcc, 4)}")



if __name__ == "__main__":
    model_name = "qwen-coder"
    # Example usage
    ori_file_list = [f'/root/sy/ptw4awi/1_validity/results/{model_name}/ori_1_raw_preds.csv',
                 f'/root/sy/ptw4awi/1_validity/results/{model_name}/ori_2_raw_preds.csv', 
                 f'/root/sy/ptw4awi/1_validity/results/{model_name}/ori_3_raw_preds.csv',
                 f'/root/sy/ptw4awi/1_validity/results/{model_name}/ori_4_raw_preds.csv',
                 f'/root/sy/ptw4awi/1_validity/results/{model_name}/ori_5_raw_preds.csv']  # Replace with your actual file paths
    ctx_file_list = [f'/root/sy/ptw4awi/1_validity/results/{model_name}/ctx_1_raw_preds.csv',
                    f'/root/sy/ptw4awi/1_validity/results/{model_name}/ctx_2_raw_preds.csv', 
                    f'/root/sy/ptw4awi/1_validity/results/{model_name}/ctx_3_raw_preds.csv',
                    f'/root/sy/ptw4awi/1_validity/results/{model_name}/ctx_4_raw_preds.csv',
                    f'/root/sy/ptw4awi/1_validity/results/{model_name}/ctx_5_raw_preds.csv']  # Replace with your actual file paths
    abs_file_list = [f'/root/sy/ptw4awi/1_validity/results/{model_name}/abs_1_raw_preds.csv',
                    f'/root/sy/ptw4awi/1_validity/results/{model_name}/abs_2_raw_preds.csv', 
                    f'/root/sy/ptw4awi/1_validity/results/{model_name}/abs_3_raw_preds.csv',
                    f'/root/sy/ptw4awi/1_validity/results/{model_name}/abs_4_raw_preds.csv',
                    f'/root/sy/ptw4awi/1_validity/results/{model_name}/abs_5_raw_preds.csv']  # Replace with your actual file paths
    print(" ==================== ori ==================== ")
    main(ori_file_list)
    print(" ==================== ctx ==================== ")
    main(ctx_file_list)
    print(" ==================== abs ==================== ")
    main(abs_file_list)