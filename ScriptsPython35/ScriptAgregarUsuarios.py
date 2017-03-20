import numpy as np
import pandas as pd


def agrega_usuarios():
    pd.options.mode.chained_assignment = None
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    #n_cols = ['new_user_id', 'movie_id', 'rating', 'unix_timestamp']
    pasta = "D:\\TCC_2\\ml-100k\\u"
    g_cols = ['id_group',  'group_set']
    groups_sim = pd.read_csv('D:\\TCC_2\\ml-100k\\groups_simi.csv', sep='\t', names=g_cols)
    groups_rand = pd.read_csv('D:\\TCC_2\\ml-100k\\groups_rand.csv', sep='\t', names=g_cols)
    for i in range(1, 6):
        train = pd.read_csv(pasta+str(i)+".base", sep='\t', names=r_cols)
        test = pd.read_csv(pasta+str(i)+".test", sep='\t', names=r_cols)
        simi_tr_csv = pd.DataFrame(columns=r_cols)
        simi_te_csv = pd.DataFrame(columns=r_cols)
        rand_tr_csv = pd.DataFrame(columns=r_cols)
        rand_te_csv = pd.DataFrame(columns=r_cols)

        for j in range(0, 3000):
            set_sim = list(eval(list(groups_sim[groups_sim.id_group == j]['group_set'])[0]))
            set_rand = list(eval(list(groups_rand[groups_sim.id_group == j]['group_set'])[0]))

            simi_tr = train[((train.user_id == set_sim[0]) | (train.user_id == set_sim[1]) | (train.user_id == set_sim[2]) | (train.user_id == set_sim[3]) | (train.user_id == set_sim[4]))]
            simi_te = test[((test.user_id == set_sim[0]) | (test.user_id == set_sim[1]) | (test.user_id == set_sim[2]) | (test.user_id == set_sim[3]) | (test.user_id == set_sim[4]))]
            rand_tr = train[((train.user_id == set_rand[0]) | (train.user_id == set_rand[1]) | (train.user_id == set_rand[2]) | (train.user_id == set_rand[3]) | (train.user_id == set_rand[4]))]
            rand_te = test[((test.user_id == set_rand[0]) | (test.user_id == set_rand[1]) | (test.user_id == set_rand[2]) | (test.user_id == set_rand[3]) | (test.user_id == set_rand[4]))]

            simi_tr["user_id"] = j
            simi_te["user_id"] = j
            rand_tr["user_id"] = j
            rand_te["user_id"] = j

            simi_tr_csv = simi_tr_csv.append(simi_tr, ignore_index=True).astype(int)
            simi_te_csv = simi_te_csv.append(simi_te, ignore_index=True).astype(int)
            rand_tr_csv = rand_tr_csv.append(rand_tr, ignore_index=True).astype(int)
            rand_te_csv = rand_te_csv.append(rand_te, ignore_index=True).astype(int)
        simi_tr_csv.to_csv("user-agreggation\\u_simi" + str(i) + ".base", sep='\t', encoding='utf8',header=False,index=False,index_label=False)
        simi_te_csv.to_csv("user-agreggation\\u_simi" + str(i) + ".test", sep='\t', encoding='utf8',header=False,index=False,index_label=False)
        rand_tr_csv.to_csv("user-agreggation\\u_rand" + str(i) + ".base", sep='\t', encoding='utf8',header=False,index=False,index_label=False)
        rand_te_csv.to_csv("user-agreggation\\u_rand" + str(i) + ".test", sep='\t', encoding='utf8',header=False,index=False,index_label=False)

if __name__ == '__main__':
    agrega_usuarios()
