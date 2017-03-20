import pandas as pd
import numpy as np
import math


def precision(list_rank,df_test):
    result_list = []
    for i in range(0,3000):
        items_on_test = set(df_test[df_test.user_id == i]['item_id'])
        items_on_pred = set(list_rank[list_rank.user_id==i].sort_values([('rating')],ascending = False)['item_id'][:5])
        intersect = items_on_test.intersection(items_on_pred)
        result_list.append(len(intersect) / len(items_on_pred))
    return np.mean(result_list)


def recall(list_rank,df_test):
    result_list = []
    for i in range(0,3000):
        items_on_test = set(df_test[df_test.user_id == i]['item_id'])
        items_on_pred = set(list_rank[list_rank.user_id==i].sort_values([('rating')],ascending = False)['item_id'][:5])
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


def ndcg(list_rank, df_test):
    result_list = []
    for i in range(0, 3000):
        items_on_test = set(df_test[df_test.user_id == i]['item_id'])
        items_on_pred = set(list_rank[list_rank.user_id==i].sort_values([('rating')],ascending = False)['item_id'][:5])
        dcg_list = []
        idcg_list = []
        for i in range(0,len(items_on_pred)):
            if(list(items_on_pred)[i] in items_on_test):
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

def eval():
    u_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    result_pre_simi = []
    result_rec_simi = []
    result_ndcg_simi = []
    result_pre_rand = []
    result_rec_rand = []
    result_ndcg_rand = []
    for i in range(1,6):
        rand_on_test = pd.read_csv('user-agreggation\\u_rand'+str(i)+'.test', sep='\t', names=u_cols)
        sim_on_test = pd.read_csv('user-agreggation\\u_simi' + str(i) + '.test', sep='\t', names=u_cols)
        agreg_sim = pd.read_csv('D:\\TCC_2\\ScriptsIronPython\\Recomendacao-agreg-user\\Positive_rec_s'+str(i)+'.res', sep='\t', names=['user_id','item_id','rating'])
        agreg_rand = pd.read_csv('D:\\TCC_2\\ScriptsIronPython\\Recomendacao-agreg-user\\Positive_rec_r'+str(i)+'.res', sep='\t', names=['user_id','item_id','rating'])
        result_pre_simi.append(precision(agreg_sim,sim_on_test))
        result_rec_simi .append(recall(agreg_sim,sim_on_test))
        result_ndcg_simi.append(ndcg(agreg_sim,sim_on_test))
        result_pre_rand.append(precision(agreg_rand,rand_on_test))
        result_rec_rand.append(recall(agreg_rand,rand_on_test))
        result_ndcg_rand.append(ndcg(agreg_rand,rand_on_test))
    result_to_file = []
    result_to_file.append('precision_s = ' + str(np.mean(result_pre_simi)))
    result_to_file.append('recall_s = ' + str(np.mean(result_rec_simi)))
    result_to_file.append('ndcg_s = ' + str(np.mean(result_ndcg_simi)))
    result_to_file.append('precision_r = ' + str(np.mean(result_pre_rand)))
    result_to_file.append('recall_r = ' + str(np.mean(result_rec_rand)))
    result_to_file.append('ndcg_r = ' + str(np.mean(result_ndcg_rand)))
    tofile_s = pd.Series(result_to_file)
    tofile_s.to_csv("UserAgreggation_Result", sep='\t', encoding='utf8')

if __name__ == '__main__':
    eval()