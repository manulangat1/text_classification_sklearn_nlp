import json
reviews = []
with open('./Books_small.json', 'r' ) as f:
    for line in f:
        review = json.loads(line)
        reviews.append((review['reviewText'],review['overall']))

print(len(reviews))
print(reviews[5][1])


###make a data class