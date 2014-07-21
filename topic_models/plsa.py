#!/usr/bin/env python

""" Probabilistic Latent Semantic Analysis """ 

from numpy import zeros, log, dot
from numpy.random import random


class plsa:
    def __init__(self, n_components=2, n_iter=5):
        self.num_topics = n_components 
        self.num_models = n_iter
        self.num_words = None
        self.num_docs = None
        self.p_w_z = None
        self.p_z_d = None
    
    
    def fit(self, X, y=None):
        """ Discover a low-dimensional semantic space from X, using EM.
        """ 
        self.num_docs, self.num_words = X.shape
        self.p_w_z, self.p_z_d = plsa_em(X, self.num_topics, self.num_models)

    
    def transform(self, X):
        pass


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.p_z_d


def plsa_em(X, num_topics, num_models,
            min_delta_l=0.0001, max_em_iter=10000):
    """ Expectation Maximization for the PLSA topic model

        Inputs:
          -X:  Input data matrix. (n_samples, n_features)
          -num_topics:   Number of topics to learn. 
          -num_models:   Number of times to run the EM algorithm.
                         Depending on initial conditions, EM may
                         converge to a local maxima, not necessarily
                         the global maxima.  For this reason, the 
                         algorithm can be ran several times, selecting
                         the best model from the set of iterations. 
          -min_delta_l:  Minimum percent change for the log-likelihood.
                         Stop when improvement falls below this threshold.
          -max_em_iter:  Maximum number of iteration for the EM algorithm.
                 
        Returns:
          -p_w_z:  Probabilities of each word, given a topic.
          -p_z_d:  Probabilities of each topic, given a document. 
    """
    X = X.astype(float)
    num_docs, num_words = X.shape
    p_w_z_best = zeros((num_words, num_topics))
    p_z_d_best = zeros((num_topics, num_docs))
    p_w_z = zeros(p_w_z_best.shape)
    p_z_d = zeros(p_z_d_best.shape)
    p_z_wd = zeros((num_topics, num_words, num_docs))
    l_best = 0

    """ P(d) or P(D=d) is just the proportion of words 
        in the corpus that belong to d, which is the 
        sum of word counts in each document, divided
        by the total sum of words.  
        
        Note:  axis=1 specifies to sum over the "n_features".
    """ 
    p_d = X.sum(axis=1)/X.sum()

    for model_iter in xrange(num_models):
        em_rounds = 0
        p_w_z[:] = random(p_w_z.shape)
        p_z_d[:] = random(p_z_d.shape)
        """ [:,None] transforms what would be an array 
            with shape (num_words,) to (num_words,1)
            so that each word's prob is divided by
            the sum of all words in the topic. 
        """
        p_w_z /= p_w_z.sum(axis=1)[:,None]  # sum(p_w_z[0, :]) ~ 1
        p_z_d /= p_z_d.sum(axis=1)[:,None]

        l_current = log_likelihood(X, p_w_z, p_z_d, p_d)
        l_old = 0
        em_iter = 0

        print '\nRound %d' % (model_iter+1)

        while (em_iter < max_em_iter) and \
              (abs(l_current-l_old) > abs(l_old)*min_delta_l):
            # Still improving and fewer than max iterations 
            l_old = l_current
            em_iter += 1

            # Run EM
            em_e_step(p_z_wd, p_w_z, p_z_d)        
            em_m_step(X, p_w_z, p_z_d, p_z_wd)
            l_current = log_likelihood(X, p_w_z, p_z_d, p_d)

        if l_current < l_best:
            p_w_z_best[:] = p_w_z  # Deep copy!
            p_z_d_best[:] = p_z_d
            l_best = l_current

    return p_w_z_best, p_z_d_best


def em_e_step(p_z_wd, p_w_z, p_z_d):
    """ Given P(w|z) and P(z|d), compute
        P(z|w,d) as...
              P(w|z)*P(z|d)
        -------------------------
        SumOver(z): P(w|z)*P(z|d)
    """
    num_words, num_topics = p_w_z.shape
    _, num_docs = p_z_d.shape
    
    for i in xrange(num_docs):
        for j in xrange(num_words):
            for k in xrange(num_topics):
                p_z_wd[k, j, i] = p_w_z[j, k] * p_z_d[k, i]

    """ Probability for all topics given each
        doc-word pair must sum to one """ 
    for i in xrange(num_docs):
        for j in xrange(num_words):
            total_prob = dot(p_w_z[j,:], p_z_d[:,i])
            if total_prob > 0:
                p_z_wd[:,j,i] /= total_prob
            else:
                p_z_wd[:,j,i] = 0


def em_m_step(X, p_w_z, p_z_d, p_z_wd, folding_in=False):
    """ Update p_w_z and p_z_d """ 
    num_topics, num_words, num_docs = p_z_wd.shape
    
    # Update p_w_z -- don't touch if folding in.
    if not folding_in:
        p_w_z[:] = 0.
        for j in xrange(num_words):
            for k in xrange(num_topics):
                for i in xrange(num_docs):
                    p_w_z[j,k] += X[i,j] * p_z_wd[k,j,i]
        
        for j in xrange(num_words):
            for k in xrange(num_topics):
                p_w_z[:,k] /= sum(p_w_z[:,k])

    # Update p_z_d
    for i in xrange(num_docs):
        for k in xrange(num_topics):
            p_z_d[k,i] = dot(X[i,:], p_z_wd[k,:,i].transpose()) \
                         / sum(X[i,:])
   

def log_likelihood(X, p_w_z, p_z_d, p_d):
    """ Compute the log likelihood of the data
        given the provided distributions 
    
        sum over all documents:
            sum over all words:
                (number of times word occurred in document 
                time the log-prob of that happening.) 
    """ 
    num_docs, num_words = X.shape
    _, num_topics = p_w_z.shape
    log_likelihood = 0

    for i in xrange(num_docs):
        for j in xrange(num_words):
            inner_sum = 0.
            for k in xrange(num_topics):
                inner_sum += p_w_z[j, k] * p_z_d[k, i]
            p_dw = inner_sum * p_d[i]
            # P(d,w) = P(w|d)*P(d) = sum_{z\inZ} P(w|z)*P(z|d)*P(d)
            if p_dw > 0:
                log_likelihood += X[i,j] * log(p_dw)

    return log_likelihood


if __name__=="__main__":
    from numpy.random import poisson
    X = poisson(1, (10, 1000))
    clf = plsa(n_components=2, n_iter=10)
    clf.fit(X)
