#!/usr/bin/env python
"""Unit tests for parser support libraries."""

from unittest import TestCase, main
from numpy import array, bincount
from numpy.random import shuffle
from collections import Counter
from biotm.misc.util import BalancedKFold

class utilTests(TestCase):
    """ Test misc utils. """

    def test_balanced_kfold(self):
        k = 10
        labels = array([1]*100 + [0]*200)
        shuffle(labels)
        for training, testing in BalancedKFold(labels, 10):
            """ For each of the folds, make sure the counts of 
                one class are equal to the counts of the other """ 
            t = bincount(labels[training])
            self.assertEqual(t[0], t[1])
            t = bincount(labels[testing])
            self.assertEqual(t[0], t[1])
        

    def test_balanced_kfold2(self):
        k = 10
        labels = array([1]*101 + [0]*200)
        # SAME EXCEPT FOR  ^^^
        shuffle(labels)
        for training, testing in BalancedKFold(labels, 10):
            """ For each of the folds, make sure the counts of 
                one class are equal to the counts of the other """ 
            t = bincount(labels[training])
            self.assertEqual(t[0], t[1])
            t = bincount(labels[testing])
            self.assertEqual(t[0], t[1])
        

if __name__ == '__main__':
    main()
