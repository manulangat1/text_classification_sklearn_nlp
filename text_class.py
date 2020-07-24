import json
from sklearn.model_selection import train_test_split
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