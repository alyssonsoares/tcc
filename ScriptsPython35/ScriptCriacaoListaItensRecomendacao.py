import pandas as pd
u_cols = ['user_id','item_id','rating','timestamp']
pasta = 'ml-100k\\'
for i in range(1,6):
    users_train = pd.read_csv(pasta + "u" + str(i) + ".base",sep='\t',names=u_cols)
    users_test = pd.read_csv(pasta + "u" + str(i) + ".test",sep='\t',names=u_cols)
    items_tr = set(users_train['item_id'].unique())
    items_te = set(users_test['item_id'].unique())
    list_recommender = items_tr.intersection(items_tr)
    list_recommender = list(list_recommender)
    result = pd.Series(list_recommender)
    result.to_csv("files-to-recommender\\items_rec_" + str(i) + ".result",sep="\t",encoding='utf8')

