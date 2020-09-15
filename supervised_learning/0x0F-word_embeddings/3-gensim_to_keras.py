#!/usr/bin/env python3
"""Gensim to keras"""

def gensim_to_keras(model):
    """
    gensim to keras function
    Args:
        model: trained gensim word2vec models
    Returns: trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=False)
