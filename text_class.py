import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import CountVectorizer
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
with open('./Books_small.json', 'r' ) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'],review['overall']))

print(len(reviews))
print(reviews[5].text)


training,test = train_test_split(reviews,test_size=0.33,random_state=42)
train_x = [ x.text for x in training ]
train_y = [ x.sentiment for x in training ]


test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]


vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

##classification 
##LinearSVM
# from sklearn import svm

# clf_svm = svm.SVC(
#     kernel='linear'
# )
# clf_svm.fit(train_x_vectors,train_y)

# clf_svm.predict(test_x_vectors)

###DECison Treee Classifier
# from sklearn.tree import DecisionTreeClassifier
# clf_dec = DecisionTreeClassifier()
# clf_dec.fit(train_x_vectors,train_y)
# clf_dec.predict(test_x_vectors)

###Naive Bayes 
# from sklearn.naive_bayes import GaussianNB 
# clf_gnb = GaussianNB
# clf_gnb.fit(train_x_vectors,train_y)
# clf_gnb.predict(test_x_vectors)

##Logistic Regression 
# from sklearn.linear_model import LogisticRegression 
# clf_log = LogisticRegression()
# clf_log.fit(train_x_vectors,train_y)
# clf_log.predict(test_x_vectors)