localDll = "D:\\TCC_2\\"
path = "D:\\TCC_2\\ScriptsPython35\\user-agreggation\\u_simi"
import clr
import csv
import heapq
clr.AddReferenceToFileAndPath(localDll+"MyMediaLite.dll")
from MyMediaLite import *
from Util import IndividualRec,UserRec

recommender_p = ItemRecommendation.UserKNN()
recommender_r = RatingPrediction.UserItemBaseline()
evaluate_p=[]
evaluate_r=[]
def recommenderPositiveOnly(train,test):
    #load the data
    train_data = IO.ItemData.Read(train)
    test_data = IO.ItemData.Read(test)
    # set up the recommender
    recommender_p.K = 20
    recommender_p.Feedback = train_data
    recommender_p.Train()
    evaluate_p.append(Eval.Items.Evaluate(recommender_p, test_data, train_data))
    return recommender_p

def recommenderRatingPrediction(train,test):
    # load the data
    train_data = IO.RatingData.Read(train)
    test_data  = IO.RatingData.Read(test)
    # set up the recommender
    recommender_r.Ratings = train_data
    recommender_r.Train()
    evaluate_r.append(Eval.Ratings.Evaluate(recommender_r, test_data))
    return recommender_r

#write result in csv file
def writeResult(array,name):
    with open(name,"wb") as csvfile:
        writer = csv.writer(csvfile,delimiter="\t",quotechar='|',quoting=csv.QUOTE_MINIMAL)
        for a in array:
            for i in a.items:
                writer.writerow([a.user,i.item,i.rating])
    
    
# make recommendation individual
for i in range(1,6):
    train = path+str(i)+".base"
    test = path+str(i)+".test"
    recommenderPositiveOnly(train,test)
    #recommenderRatingPrediction(train,test)
    #individualRecs_r=[]
    individualRecs_p=[]
    for user in range(0,3000):
        list_r = []
        list_p = []
        with open("D:\\TCC_2\\files-to-recommender\\items_rec_"+str(i)+".result","rb") as csvfile:
            reader = csv.reader(csvfile,delimiter='\t',quotechar="|")
            for row in reader:
               # list_r.append(IndividualRec(int(row[1]),recommender_r.Predict(user,int(row[1])),0))
                list_p.append(IndividualRec(int(row[1]),recommender_p.Predict(user,int(row[1])),0))
        #individualRecs_r.append(UserRec(user,list_r))
        individualRecs_p.append(UserRec(user,list_p))
    #writeResult(individualRecs_r,"Recomendacao-agreg-user\\Rating_rec_s"+str(i)+".res")
    writeResult(individualRecs_p,"Recomendacao-agreg-user\\Positive_rec_s"+str(i)+".res")

# measure the accuracy on the test data set
with open("measures\\measureResult","wb") as csvfile:
    writer = csv.writer(csvfile,delimiter="\t",quotechar='|',quoting=csv.QUOTE_MINIMAL)
    for ev in evaluate_p:
        i=0
        writer.writerow(["fold_pos_s"+str(i),ev])
        i+=1
    #for ev in evaluate_r:
    #    i=0
    #    writer.writerow(["fold_rat_s"+str(i),ev])
    #    i+=1




