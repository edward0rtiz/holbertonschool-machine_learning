#!/usr/bin/env python3
import numpy as np
uni_bleu = __import__('0-uni_bleu').uni_bleu

"""references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))"""
references = [["hello", "there", "my", "friend"]]
sentence = ["hello", "there", "comrade"]

print(np.round(uni_bleu(references, sentence), 10))