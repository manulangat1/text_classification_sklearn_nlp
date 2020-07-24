import json
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
###make a data class
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
class Review:
    def __init__(self,text,score):
        self.text=text
        self.score=score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE
reviews = []
class ReviewContainer:
    def __init__(self,reviews):
        self.reviews = reviews
    def get_text(self):
       return [ x.text for x in self.reviews]
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
    def evenly_distribute(self):
        negative = list(filter(lambda x:x.sentiment == Sentiment.NEGATIVE,self.reviews))
        print(negative[0].sentiment)
        positive = list(filter(lambda x:x.sentiment == Sentiment.POSITIVE,self.reviews))
        print(len(negative),len(positive))
        positive_shrink = positive[:len(negative)]
        self.reviews = negative + positive_shrink
        random.shuffle(self.reviews)
        print(len(reviews))

with open('./Books_small_10000.json',  'r' ) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'],review['overall']))

# print(len(reviews))
# print(reviews[5].text)


training,test = train_test_split(reviews,test_size=0.33,random_state=42)

train_cont = ReviewContainer(training)
test_cont = ReviewContainer(test)

train_x = train_cont.get_text()
train_y = train_cont.get_sentiment()
test_x = test_cont.get_text()
test_y = test_cont.get_sentiment()

print(train_y[0])
# vectorizer = CountVectorizer()
# train_x_vectors = vectorizer.fit_transform(train_x)
# test_x_vectors = vectorizer.transform(test_x)

# ##classification 
# ##LinearSVM
# from sklearn import svm

# clf_svm = svm.SVC(
#     kernel='linear'
# )
# clf_svm.fit(train_x_vectors,train_y)

# print(clf_svm.predict(test_x_vectors[0]))
# print(test_y[0])

# ###DECison Treee Classifier
# from sklearn.tree import DecisionTreeClassifier
# clf_dec = DecisionTreeClassifier()
# clf_dec.fit(train_x_vectors,train_y)
# print(clf_dec.predict(test_x_vectors[0]))
# print(test_y[0])

# ###Naive Bayes 
# from sklearn.naive_bayes import GaussianNB 
# clf_gnb = GaussianNB()
# clf_gnb.fit(train_x_vectors.todense(),train_y)
# print(clf_gnb.predict(test_x_vectors.todense()[0]))
# print(test_y[0])

# ##Logistic Regression 
# from sklearn.linear_model import LogisticRegression 
# clf_log = LogisticRegression()
# clf_log.fit(train_x_vectors,train_y)
# print(clf_log.predict(test_x_vectors[0]))
# print(test_y[0])
# ##Mean accureacy
# # print(clf_svm.score(test_x_vectors,test_y))
# # print(clf_dec.score(test_x_vectors,test_y))
# # print(clf_gnb.score(test_x_vectors.todense(),test_y))
# # print(clf_log.score(test_x_vectors,test_y))

# from sklearn.metrics import f1_score
# print(f1_score(test_y,clf_svm.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE]))
# print(f1_score(test_y,clf_dec.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE]))
# print(f1_score(test_y,clf_gnb.predict(test_x_vectors.todense()),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE]))
# print(f1_score(test_y,clf_log.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE]))

# print(train_y.count(Sentiment.NEGATIVE))