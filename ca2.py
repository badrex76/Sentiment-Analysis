import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import string
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk import classify
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
import nltk.metrics
import collections
from collections import OrderedDict



documents=[]
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))




all_words = [word.lower() for word in movie_reviews.words()]


stopwords = stopwords.words('english')


words_clean = []
for word in all_words:
    if word not in stopwords and word not in string.punctuation:
        words_clean.append(word)


#فرکانش کلمات top-N words feature
words_frequency = FreqDist(words_clean)

# get 2000 frequently occuring words most
most_frequency_word = words_frequency.most_common(2000)


word_features = [item[0] for item in most_frequency_word]


def document_features(document):
    # "set" function will remove repeated/duplicate tokens in the given list
    document_words = set(document)
    features = {}
    for word in word_features:
        features['%s' % word] = (word in document_words)
    return features

feature_set = [(document_features(doc), category) for (doc, category) in documents]

NUM_SPLITS = 5
kfold = KFold(n_splits=NUM_SPLITS)
split_data = kfold.split(feature_set)




for train, test in split_data:
     output_train = []
     output_test = []


     for i in train:
         output_train.append(feature_set[i])


     for i in test:
         output_test.append(feature_set[i])



classifier = NaiveBayesClassifier.train(output_train)
accuracy = classify.accuracy(classifier, output_test)

print('accuracy top-N:',accuracy)


from nltk.metrics.scores import (precision, recall)

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(output_test):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)


p=precision(refsets['pos'], testsets['pos'])
print('precision top-N:',p)
r=recall(refsets['pos'], testsets['pos'])
print('recall top-N:',r)
#
######bow
pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)


def bag_of_words(words):
    words_clean = []

    for word in words:
        word = word.lower()
        if word not in stopwords and word not in string.punctuation:
            words_clean.append(word)

    words_dictionary = dict([word, True] for word in words_clean)

    return words_dictionary


# print (bag_of_words(['the', 'the', 'good', 'bad', 'the', 'good']))
# '''
# Output:
#
# {'bad': True, 'good': True}
# '''

pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_words(words), 'pos'))

# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_words(words), 'neg'))


test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
train_set = pos_reviews_set[200:] + neg_reviews_set[200:]

classifier = NaiveBayesClassifier.train(train_set)

accuracy = classify.accuracy(classifier, test_set)
print('accuracy unigram :' ,accuracy)

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)


p=precision(refsets['pos'], testsets['pos'])
print('precision unigram :' ,p)
r=recall(refsets['pos'], testsets['pos'])
print('recall unigram :' ,r)


#####bigram


from nltk import ngrams


def bag_of_ngrams(words,n=2):
    words_clean = []

    for word in words:
        word = word.lower()
        if word not in stopwords and word not in string.punctuation:
            words_clean.append(word)

    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)
    return words_dictionary


pos_bi_reviews_set = []
for words in pos_reviews:
    pos_bi_reviews_set.append((bag_of_ngrams(words), 'pos'))

# negative reviews feature set
neg_bi_reviews_set = []
for words in neg_reviews:
    neg_bi_reviews_set.append((bag_of_ngrams(words), 'neg'))



test_bi_set = pos_bi_reviews_set[:200] + neg_bi_reviews_set[:200]
train_bi_set = pos_bi_reviews_set[200:] + neg_bi_reviews_set[200:]

classifier = NaiveBayesClassifier.train(train_bi_set)

accuracy = classify.accuracy(classifier, test_bi_set)
print('accuracy bigram:',accuracy)


refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(test_bi_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)


p=precision(refsets['pos'], testsets['pos'])
print('precision bigram:',p)
r=recall(refsets['pos'], testsets['pos'])
print('recall bigram:',r)
