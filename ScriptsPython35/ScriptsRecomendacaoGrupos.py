import pandas as pd
import numpy as np
import heapq
u_cols = ['user_id','item_id','predict']
g_cols = ['id_group','group_set']
predict_p = pd.DataFrame()
predict_r = pd.DataFrame()
result_a = []
result_l = []
groups_simi = pd.read_csv('D:\TCC_2\\ml-100k\\groups_simi.csv',sep='\t',names=g_cols)
groups_rand = pd.read_csv('D:\TCC_2\\ml-100k\\groups_rand.csv',sep='\t',names=g_cols)
groups_s = []
groups_r = []


#Generate AVG ranked List
def RankedListAvg(group):
    result_a.append(predict_p[predict_p.user_id.isin(group)].groupby('item_id').agg({'predict':np.mean}).sort_values([('predict')],ascending=False)[:5])


#Generate LM ranked List
def RankedListLM(group):
    result_l.append(predict_p[predict_p.user_id.isin(group)].groupby('item_id').agg({'predict':np.min}).sort_values([('predict')],ascending=False)[:5])

def getListRank(groups):
    for j in range(0,3000):
        RankedListAvg(groups[j])
        RankedListLM(groups[j])

def printFile(pdSeries, name):
    tofile_s = pd.Series(pdSeries)
    tofile_s.to_csv(name, sep='\t', encoding='utf8')


groups = groups_simi['group_set']
for g in groups:
    groups_s.append(list(eval(g)))
groups = groups_rand['group_set']
for g in groups:
    groups_r.append(list(eval(g)))
for i in range(1,6):
    predict_p = pd.read_csv('D:\\TCC_2\\individual-rec\\Positive_rec' + str(i) + '.res' ,sep='\t',names=u_cols)
    getListRank(groups_s)
    printFile(result_a, "D:\\TCC_2\\group-rec-simi\\Positive_RankedList_AVG_fold_" + str(i) + ".result")
    printFile(result_l, "D:\\TCC_2\\group-rec-simi\\Positive_RankedList_LM_fold_" + str(i) + ".result")
    result_a = []
    result_l = []
    getListRank(groups_r)
    printFile(result_a, "D:\\TCC_2\\group-rec-rand\\Positive_RankedList_AVG_fold_" + str(i) + ".result")
    printFile(result_l, "D:\\TCC_2\\group-rec-rand\\Positive_RankedList_LM_fold_" + str(i) + ".result")
    result_a = []
    result_l = []

