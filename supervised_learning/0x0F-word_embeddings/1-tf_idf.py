#!/usr/bin/env python3
"""tf_idf technique"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    tf_idf
    Args:
        sentences: list of sentences to analize
        vocab: list of the vocabulary words to use for the analysis
    Returns: embeddings, features
    """
    if vocab is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embedding = X.toarray()

    return embedding, vocab