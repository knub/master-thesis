from knub.thesis import stop_words
from collections import defaultdict


def remove_stop_words(documents):
    return [[word for word in document.lower().split() if word not in stop_words]
             for document in documents]


def remove_rare_words(documents):
    frequency = defaultdict(int)
    for text in documents:
        for token in text:
            frequency[token] += 1
    return [[token for token in text if frequency[token] > 1]
            for text in documents]