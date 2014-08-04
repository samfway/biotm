#!/usr/bin/env python
"""Unit tests for expectation maximization."""

from biotm.topic_models.lda.lda import lda

from unittest import TestCase, main
from numpy import array, zeros
from numpy.random import random

class ldaTests(TestCase):
    """ Basic LDA sanity checks """
    def setUp(self):
        self.num_topics = 10
        self.num_docs = 20
        self.num_words = 100


    def test_lda_dimensions(self):
        """ Make sure dimensions line up after 
            transforming a new document with LDA
        """ 
        new_docs = 15
        X = random((self.num_docs, self.num_words))
        Xn = random((new_docs, self.num_words))
        tm = lda(n_components=self.num_topics)
        tm.fit(X)
        Xn_txd = tm.transform(Xn)
        Xn_docs, Xn_words = Xn_txd.shape

        self.assertEqual(Xn_docs, new_docs)
        self.assertEqual(Xn_words, self.num_topics)


    def test_fit_transform(self):
        """ Make sure fit_transform returns the
            dimensions that it should be.  
        """ 
        X = random((self.num_docs, self.num_words))
        tm = lda(n_components=self.num_topics)
        Xt = tm.fit_transform(X)
        Xt_docs, Xt_words = Xt.shape

        self.assertEqual(Xt_docs, self.num_docs)
        self.assertEqual(Xt_words, self.num_topics)


if __name__ == '__main__':
    main()
