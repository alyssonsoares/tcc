import pandas as pd
import numpy as np
import math
def precision(groups, list_rank,df_test):
    result_list = []
    for i in range(0,len(groups)):
        items_on_test = set(df_test[df_test.user_id.isin(groups[i])]['item_id'])
        items_on_pred = set(list_rank[i]['item_id'])
        values = []
        for j in items_on_pred:
            values.append(eval(j))
        items_on_pred = set(values)
        intersect = items_on_test.intersection(items_on_pred)
        result_list.append(len(intersect) / len(items_on_pred))
    return np.mean(result_list)

def recall(groups, list_rank,df_test):
    result_list = []
    for i in range(0,len(groups)):
        items_on_test = set(df_test[df_test.user_id.isin(groups[i])]['item_id'])
        items_on_pred = set(list_rank[i]['item_id'])
        values = []
        for j in items_on_pred:
            values.append(eval(j))
        items_on_pred = set(values)
        intersect = items_on_test.intersection(items_on_pred)
        if(len(items_on_test) == 0):
            result_list.append(0)
        else:
            result_list.append(len(intersect) / len(items_on_test))
    return np.mean(result_list)

def dcg(r_list,N):
    result = []
    ranked_list = []
    ranked_list = r_list
    for k in range(0,N):
        if(k == 0):
            result.append(ranked_list[k])
        else:
            if(ranked_list[k] == 0):
                result.append(0)
            else:
                result.append(ranked_list[k] / math.log2(k + 1))
    return np.sum(result)
def ndcg(groups, list_rank,df_test):
    result_list = []
    for i in range(0,len(groups)):
        items_on_test = set(df_test[df_test.user_id.isin(groups[i])]['item_id'])
        items_on_pred = set(list_rank[i]['item_id'])
        dcg_list = []
        idcg_list = []
        for i in range(0,len(items_on_pred)):
            if(eval(list(items_on_pred)[i]) in items_on_test):
                val = 1
            else:
                val = 0
            dcg_list.append(val)
            idcg_list.append(val)
        N = len(dcg_list)
        idcg_list.sort(reverse=True)
        idcg = dcg(idcg_list,N)
        if(idcg == 0):
            result_list.append(0)
        else:
            result_list.append(dcg(dcg_list,N) / idcg)
    return np.mean(result_list)

 

def evaluate():
    group_rec_simi = "D:\\TCC_2\\group-rec-simi\\"
    group_rec_rand = "D:\\TCC_2\\group-rec-rand\\"
    u_cols = ['user_id','item_id','rating','timestamp']
    g_cols = ['id_group','group_set']
    algoritmo_recInd = "Positive"
    aggregation = "AVG"
    groups_simi = pd.read_csv('ml-100k\\groups_simi.csv',sep='\t',names=g_cols)
    groups_rand = pd.read_csv('ml-100k\\groups_rand.csv',sep='\t',names=g_cols)
    group_s = []
    group_r = []
    groups = groups_simi['group_set']
    for g in groups:
        group_s.append(list(eval(g)))
    groups = groups_rand['group_set']
    for g in groups:
        group_r.append(list(eval(g)))
    list_rank_simi = []
    list_rank_rand = []
    result_pre_simi = []
    result_rec_simi = []
    result_ndcg_simi = []
    result_pre_rand = []
    result_rec_rand = []
    result_ndcg_rand = []
    for i in range(1,6):
        df_items_test = pd.read_csv("ml-100k\\u" + str(i) + ".test",sep='\t',names=u_cols)
        df_group_simi = pd.read_csv(group_rec_simi + algoritmo_recInd + "_RankedList_" + aggregation + "_fold_" + str(i) + ".result",sep='\t',names=['group_id','ranked_list'],encoding='utf8')
        df_group_rand = pd.read_csv(group_rec_rand + algoritmo_recInd + "_RankedList_" + aggregation + "_fold_" + str(i) + ".result",sep='\t',names=['group_id','ranked_list'],encoding='utf8')
        str_ranked_simi = df_group_simi['ranked_list']
        str_ranked_rand = df_group_rand['ranked_list']
        for st in str_ranked_simi:
            vector_str = st.split('\r\n')
            column2 = vector_str[0].split('     ')[1]
            column1 = vector_str[1].split('     ')[0]
            values = []
            for vec in vector_str[2:7]:
                values.append(vec.split('      '))
            list_rank_simi.append(pd.DataFrame(values,columns=[column1,column2]))
        for sts in str_ranked_rand:
            vector_str = st.split('\r\n')
            column2 = vector_str[0].split('     ')[1]
            column1 = vector_str[1].split('     ')[0]
            values = []
            for vec in vector_str[2:7]:
                values.append(vec.split('      '))
            list_rank_rand.append(pd.DataFrame(values,columns=[column1,column2]))
        #Call evaluate metrics
        result_pre_simi.append(precision(group_s,list_rank_simi,df_items_test))
        result_rec_simi.append(recall(group_s,list_rank_simi,df_items_test))
        result_ndcg_simi.append(ndcg(group_s,list_rank_simi,df_items_test))
        result_pre_rand.append(precision(group_r,list_rank_rand,df_items_test))
        result_rec_rand.append(recall(group_r,list_rank_rand,df_items_test))
        result_ndcg_rand.append(ndcg(group_r,list_rank_rand,df_items_test))
    result_to_file = []
    result_to_file.append(algoritmo_recInd + "_" + aggregation + "_simi_precision = " + str(np.mean(result_pre_simi)))
    result_to_file.append(algoritmo_recInd + "_" + aggregation + "_simi_recall = " + str(np.mean(result_rec_simi)))
    result_to_file.append(algoritmo_recInd + "_" + aggregation + "_simi_ndcg = " + str(np.mean(result_ndcg_simi)))
    result_to_file.append(algoritmo_recInd + "_" + aggregation + "_rand_precision = " + str(np.mean(result_pre_rand)))
    result_to_file.append(algoritmo_recInd + "_" + aggregation + "_rand_recall = " + str(np.mean(result_rec_rand)))
    result_to_file.append(algoritmo_recInd + "_" + aggregation + "_rand_ndcg = " + str(np.mean(result_ndcg_rand)))
    tofile_s = pd.Series(result_to_file)
    tofile_s.to_csv(algoritmo_recInd + "_" + aggregation + "_group-measures.result",sep='\t',encoding='utf8')
if __name__ == '__main__':
    evaluate()
