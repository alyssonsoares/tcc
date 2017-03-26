#This files transform movieLens Dataset in format for LDA.
import pandas as pd

def createdataset(dataframe):
    dataset = []
    for j in range(1, len(dataframe["user_id"].unique())):
        df = dataframe[dataframe.user_id == j]
        document = ''
        for row in df.itertuples():
            item = str(row.item_id) + ' ' + str(row.rate) + ' '
            document += item
        dataset.append(document)
    return dataset

def writeBackArq(list,name):
    df = pd.DataFrame(list)
    df.to_csv('lda-format-files\\'+name,header=False,index=False,index_label=False)

if __name__ == '__main__':
    name_cols = ['user_id', 'item_id', 'rate', 'timestamp']
    caminho = "D:\\TCC_2\\ml-100k\\u"
    for i in range(1, 6):
        treino_df = pd.read_csv(caminho+str(i)+'.base',sep='\t',names=name_cols)
        teste_df = pd.read_csv(caminho+str(i)+'.test', sep='\t', names=name_cols)
        writeBackArq(createdataset(teste_df),'teste'+str(i)+'.txt')
        writeBackArq(createdataset(treino_df),'treino'+str(i)+'.txt')

