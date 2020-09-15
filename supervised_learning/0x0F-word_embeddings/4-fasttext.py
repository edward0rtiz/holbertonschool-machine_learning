#!/usr/bin/env python3
"""fasttest technique"""

import gensim


def fasttext_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    fasttext using gensim
    Args:
        sentences: list of sentences to be trained on
        size: dimensionality of the embedding layer
        min_count: the minimum number of occurrences of a
                    word for use in training
        window: maximum distance between the current and
                predicted word within a sentence
        negative: the size of negative sampling
        cbow: boolean to determine the training type; True
              is for CBOW; False is for Skip-gram
        iterations:  number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model
    Returns: the trained model
    """
    model = gensim.models.FastText(sentences, min_count=min_count,
                                   iter=iterations, size=size,
                                   window=window, negative=negative,
                                   seed=seed, sg=cbow, workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.iter)

    return model