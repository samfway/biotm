#!/usr/bin/env python

from sklearn.decomposition import TruncatedSVD, KernelPCA, FastICA, MiniBatchDictionaryLearning, NMF
from biotm.topic_models.plsa.plsa import plsa
from biotm.topic_models.r_lda.slda import slda, lda
from biotm.topic_models.lda.lda import lda


def get_methods(methods_string):
    """ Parse methods string and build list of technique tuples.
        Each tuple is (technique, technique_name) where
        technique is a scikit-learn like transformation class, and
        technique_name is a string to be used for file naming and such.  
    """
    valid_techniques = {'SVD': TruncatedSVD,
                        'NMF': NMF,
                        'PLSA': plsa,
                        'LDA': lda,
                        'SLDA': slda}

    techniques = []
    for t in methods_string.split(','):
        t = t.strip().upper()
        if t in valid_techniques:
            techniques.append((valid_techniques[t], t))
        else:
            raise ValueError('Invalid dimredux technique!')
    return techniques

