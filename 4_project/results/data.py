import pandas as pd
def calculatemodel(model_name):
    projects = ['collections', 'net', 'mavendp', 'dbcp', 'fileupload', 'configuration', 'codec', 'bcel', 'pool', 'digester']
    projects = sorted(projects)
    precision = []
    recall = []
    f1 = []
    auc = []
    mcc = []
    for project in projects:
        path = model_name+'/rq44/{}_res.txt'.format(project)
        f = open(path,'r')
        lines = f.read().splitlines()
        precision.append(lines[4].split('=')[1])
        recall.append(lines[5].split('=')[1])
        f1.append(lines[2].split('=')[1])
        auc.append(lines[1].split('=')[1])
        mcc.append(lines[3].split('=')[1])
    df = pd.DataFrame()
    df['projects'] = projects
    df['precision'] = precision
    df['recall'] = recall
    df['f1'] = f1
    df['mcc'] = mcc
    df['auc'] = auc
    df.to_csv('results/rq44/{}.csv'.format(model_name))
if __name__ == '__main__':
    calculatemodel('codet5')
    calculatemodel('codebert')
    calculatemodel('gpt2')
    calculatemodel('graphcodebert')
    calculatemodel('unixcoder')
