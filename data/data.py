import os

import pandas as pd

def split(data_type):
    df = pd.read_csv('raw/awi_{}.csv'.format(data_type))
    rq1_source = df['warning_method'].tolist()
    rq1_target = df['final_label'].tolist()
    rq1_target = [1 if t =='TP' else 0 for t in rq1_target]
    rq1_df = pd.DataFrame()
    rq1_df['source'] = rq1_source
    rq1_df['target'] = rq1_target
    rq1_df.to_csv('awi_{}.csv'.format(data_type),index=False)

    rq2_source = df['warning_line'].tolist()
    rq2_target = df['final_label'].tolist()
    rq2_target = [1 if t == 'TP' else 0 for t in rq2_target]
    rq2_df = pd.DataFrame()
    rq2_df['source'] = rq2_source
    rq2_df['target'] = rq2_target
    rq2_df.to_csv('awi_context_{}.csv'.format(data_type),index=False)

    rq2_source = df['warning_abstract_method'].tolist()
    rq2_target = df['final_label'].tolist()
    rq2_target = [1 if t == 'TP' else 0 for t in rq2_target]
    rq2_df = pd.DataFrame()
    rq2_df['source'] = rq2_source
    rq2_df['target'] = rq2_target
    rq2_df.to_csv('awi_abstract_{}.csv'.format(data_type),index=False)
def getprojectdata():
    df = pd.read_csv('raw/rawrq4.csv')
    all_projects = set(df['project'].tolist())
    print(all_projects)
    rq4_source = df['warning_method'].tolist()
    rq4_target = df['final_label'].tolist()
    rq4_project = df['project'].tolist()
    rq4_target = [1 if t == 'TP' else 0 for t in rq4_target]
    for pre_project in all_projects:
        tem_train_source = []
        tem_train_target = []
        tem_test_source = []
        tem_test_target = []
        for s,t,p in zip(rq4_source,rq4_target,rq4_project):
            if p ==pre_project:
                tem_test_source.append(s)
                tem_test_target.append(t)
            else:
                tem_train_source.append(s)
                tem_train_target.append(t)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        df_train['source'] = tem_train_source
        df_train['target'] = tem_train_target
        df_test['source'] = tem_test_source
        df_test['target'] = tem_test_target
        output_dir = 'rq4/{}'.format(pre_project)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_train.to_csv(output_dir+'/awi_train.csv')
        df_test.to_csv(output_dir + '/awi_test.csv')
def getrq42data():
    df = pd.read_csv('raw/rq4extension.csv')
    all_projects = set(df['project'].tolist())
    projects = df['project'].tolist()
    source = df['warning_method'].tolist()
    target = df['final_label'].tolist()
    target =[1 if t == 'TP' else 0 for t in target]
    flag_rq4 = df['flag4rq4'].tolist()
    for project in all_projects:
        train_p_source = []
        train_p_target = []
        test_p_source = []
        test_p_target = []
        for p,s,t,f in zip(projects,source,target,flag_rq4):
            if p==project:
                if f =='training':
                    train_p_source.append(s)
                    train_p_target.append(t)
                else:
                    test_p_source.append(s)
                    test_p_target.append(t)
        output_dir = 'rq42/{}'.format(project)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        df_train['source'] = train_p_source
        df_train['target'] = train_p_target
        df_test['source'] = test_p_source
        df_test['target'] = test_p_target
        df_train.to_csv(output_dir+'/awi_train.csv')
        df_test.to_csv(output_dir + '/awi_test.csv')
def getrq43data():
    df = pd.read_csv('raw/rq4extension.csv')
    all_projects = set(df['project'].tolist())
    projects = df['project'].tolist()
    source = df['warning_method'].tolist()
    target = df['final_label'].tolist()
    target =[1 if t == 'TP' else 0 for t in target]
    flag_rq4 = df['flag4rq4'].tolist()
    for project in all_projects:
        train_p_source = []
        train_p_target = []
        test_p_source = []
        test_p_target = []
        for p,s,t,f in zip(projects,source,target,flag_rq4):
            if p==project:
                if f =='training':
                    train_p_source.append(s)
                    train_p_target.append(t)
                else:
                    test_p_source.append(s)
                    test_p_target.append(t)
            else:
                train_p_source.append(s)
                train_p_target.append(t)
        output_dir = 'rq43/{}'.format(project)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        df_train['source'] = train_p_source
        df_train['target'] = train_p_target
        df_test['source'] = test_p_source
        df_test['target'] = test_p_target
        df_train.to_csv(output_dir+'/awi_train.csv')
        df_test.to_csv(output_dir + '/awi_test.csv')
if __name__ == '__main__':
    # split('train')
    # split('val')
    # split('test')
    # getprojectdata()
    getrq43data()