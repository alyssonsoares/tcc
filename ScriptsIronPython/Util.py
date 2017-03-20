class IndividualRec(object):
    def __init__(self,item,rating,relevance):
        self.item = item
        self.rating = rating
        self.relevance = relevance

class UserRec(object):
    def __init__(self,user,items):
        self.user=user
        self.items=items
