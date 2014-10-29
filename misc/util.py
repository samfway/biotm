#!/usr/bin/env python

from __future__ import division

__author__ = "Sam Way"
__copyright__ = "Copyright 2014, The Clauset Lab"
__license__ = "BSD"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

import warnings
from numpy import array, asarray, unique, bincount, min, floor, zeros
from numpy.random import shuffle, permutation
from sklearn.cross_validation import StratifiedKFold

def BalancedKFold(y, n_folds=3, indices=None, shuffle=False, random_state=None):
    """ Return class-balanced cross validation folds """ 
    y = asarray(y)
    n_samples = y.shape[0]
    unique_labels, y_inv = unique(y, return_inverse=True)
    n_classes = len(unique_labels)
    label_counts = bincount(y_inv)
    min_labels = min(label_counts)

    test_per_fold = floor(min_labels/n_folds)
    total_test = test_per_fold * n_classes
    train_per_fold = test_per_fold * (n_folds-1)
    total_train = train_per_fold * n_classes

    if train_per_fold < 1:
        raise ValueError("The least populated class has too few samples (%d) to "
                         "use %d-fold cross validation!" % (min_labels, n_folds))

    # Peform regular, stratified cross validation, but subsample all class
    # labels to even depth
    folds = []
    for (training, testing) in StratifiedKFold(y_inv, n_folds):
        train = []
        test = [] 
        training = permutation(training)
        testing = permutation(testing)

        saved = 0
        counts = zeros(n_classes)
        for i in training:
            if counts[y_inv[i]] < train_per_fold:
                train.append(i)
                counts[y_inv[i]] += 1
                saved += 1
                if saved >= total_train:
                    break

        saved = 0
        counts = zeros(n_classes)
        for i in testing:
            if counts[y_inv[i]] < test_per_fold:
                test.append(i)
                counts[y_inv[i]] += 1
                saved += 1
                if saved >= total_test:
                    break

        folds.append((asarray(train), asarray(test)))

    return folds 
   
    ''' 

        yield (asarray(train), asarray(test))
    '''
