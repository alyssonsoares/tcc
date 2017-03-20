import pandas as pd
from random import randint
u_cols = ['user_id','age','sex','occupation','zip_code']
users = pd.read_csv('ml-100k\\u.user',sep="|",names=u_cols,encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k\\u.data',sep='\t',names=r_cols,encoding='latin-1')

u_user = ratings['user_id'].unique()

setsomovies = []
for u in u_user :
    setsomovies.insert(u - 1, set(ratings[ratings.user_id == u]['movie_id']))

#print(setsomovies[0])
#print(set(ratings[ratings.user_id == 1]['movie_id']))
groups = []
while(len(groups) < 3000):
    G = set()
    i = randint(1,943)
    #for i in range(1,944):
    G.add(i)
    while(len(G) < 5):
        j = randint(1,943)
        #if(len(setsomovies[i-1].intersection(setsomovies[j-1]))>=5 & i!=j):
        G.add(j)
    if(G not in groups):
        groups.append(G)

result = pd.Series(groups)

result.to_csv("ml-100k\\groups_rand.csv",sep="\t",encoding='utf8')
