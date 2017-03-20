import pandas as pd
import numpy as np
import heapq
u_cols = ['user_id','item_id','predict']
g_cols = ['id_group','group_set']
predict_p = pd.DataFrame()
predict_r = pd.DataFrame()
result_p = []
result_r = []
groups_simi = pd.read_csv('ml-100k\\groups_simi.csv',sep='\t',names=g_cols)
groups_rand = pd.read_csv('ml-100k\\groups_rand.csv',sep='\t',names=g_cols)
groups_s = []
groups_r = []
#Generate AVG ranked List
#def RankedListAvg(group):
#    result_p.append(predict_p[((predict_p.user_id == group[0]) | (predict_p.user_id == group[1]) | (predict_p.user_id == group[2]) | (predict_p.user_id == group[3]) | (predict_p.user_id == group[4]))].groupby('item_id').agg({'predict':np.mean}).sort_values([('predict')],ascending=False)[:5])
#    result_r.append(predict_r[((predict_p.user_id == group[0]) | (predict_p.user_id == group[1]) | (predict_p.user_id == group[2]) | (predict_p.user_id == group[3]) | (predict_p.user_id == group[4]))].groupby('item_id').agg({'predict':np.mean}).sort_values([('predict')],ascending=False)[:5])
#Generate LM ranked List
def RankedListLM(group):
    result_p.append(predict_p[((predict_p.user_id == group[0]) | (predict_p.user_id == group[1]) | (predict_p.user_id == group[2]) | (predict_p.user_id == group[3]) | (predict_p.user_id == group[4]))].groupby('item_id').agg({'predict':np.min}).sort_values([('predict')],ascending=False)[:5])
    result_r.append(predict_r[((predict_p.user_id == group[0]) | (predict_p.user_id == group[1]) | (predict_p.user_id == group[2]) | (predict_p.user_id == group[3]) | (predict_p.user_id == group[4]))].groupby('item_id').agg({'predict':np.min}).sort_values([('predict')],ascending=False)[:5])

groups = groups_simi['group_set']
for g in groups:
    groups_s.append(list(eval(g)))
groups = groups_rand['group_set']
for g in groups:
    groups_r.append(list(eval(g)))
for i in range(1,6):
    predict_p = pd.read_csv('individual-rec\\Positive_rec' + str(i) + '.res' ,sep='\t',names=u_cols)
    predict_r = pd.read_csv('individual-rec\\Rating_rec' + str(i) + '.res' ,sep='\t',names=u_cols)
    for j in range(0,3000):
        #RankedListAvg(groups_s[j])
        RankedListLM(groups_s[j])
        print("similar " + str(j))
    tofile_s = pd.Series(result_p)
    tofile_s.to_csv("group-rec-simi\\Positive_RankedList_LM_fold_" + str(i) + ".result",sep='\t',encoding='utf8')
    tofile_s = pd.Series(result_r)
    tofile_s.to_csv("group-rec-simi\\Rating_RankedList_LM_fold_" + str(i) + ".result",sep='\t',encoding='utf8')
    result_p = []
    result_r = []
    for k in range(0,3000):
        #RankedListAvg(groups_r[k])
        RankedListLM(groups_r[k])
        print("random " + str(k))
    tofile_s = pd.Series(result_p)
    tofile_s.to_csv("group-rec-rand\\Positive_RankedList_LM_fold_" + str(i) + ".result",sep='\t',encoding='utf8')
    tofile_s = pd.Series(result_r)
    tofile_s.to_csv("group-rec-rand\\Rating_RankedList_LM_fold_" + str(i) + ".result",sep='\t',encoding='utf8')
    result_p = []
    result_r = []
