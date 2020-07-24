import json

###make a data class
class Review:
    def __init__(self,text,score):
        self.text=text
        self.score=score
reviews = []
with open('./Books_small.json', 'r' ) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'],review['overall']))

print(len(reviews))
print(reviews[5].text)


