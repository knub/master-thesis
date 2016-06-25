from gensim.models import Word2Vec

def sentences():
    yield 'I am awesome in the house to be'
    yield 'You are awesome too in the awesome'
    yield 'I like you awesome awesome'

model = Word2Vec(sentences(), size=100, window=5, min_count=0, workers=1)