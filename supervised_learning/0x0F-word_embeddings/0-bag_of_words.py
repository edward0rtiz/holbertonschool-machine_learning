#!/usr/bin/env python3
"""Bag of words"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    bag of words function
    Args:
        sentences: list of sentences to analize
        vocab: list of the vocabulary words to use for the analysis
    Returns: embeddings, features
    """
    if vocab is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = CountVectorizer(vocbulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embedding = X.toarray()

    return embedding, vocab
