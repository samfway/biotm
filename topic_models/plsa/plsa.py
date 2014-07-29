#!/usr/bin/env python

""" Probabilistic Latent Semantic Analysis """ 

from numpy import zeros, zeros_like, log, dot
from numpy.random import random

try:
    import biotm.topic_models.plsa.em.plsa_em as em_c
    USE_C = True
except:
    USE_C = False


class plsa:
    """ 
          -n_components: Number of topics to learn. 
          -n_iter:       Number of times to run the EM algorithm.
                         Depending on initial conditions, EM may
                         converge to a local maxima, not necessarily
                         the global maxima.  For this reason, the 
                         algorithm can be ran several times, selecting
                         the best model from the set of iterations. 
    """ 


    def __init__(self, n_components=2, n_iter=5):
        self.num_topics = n_components 
        self.n_iter = n_iter
        self.num_words = None
        self.num_docs = None
        self.p_w_z = None
        self.p_z_d = None
    
    
    def fit(self, X, y=None):
        """ Discover a low-dimensional semantic space from X, using EM.
        """ 
        self.num_docs, self.num_words = X.shape
        p_w_z = zeros((self.num_words, self.num_topics))
        p_z_d = zeros((self.num_topics, self.num_docs))
        self.p_w_z = zeros_like(p_w_z)
        self.p_z_d = zeros_like(p_z_d)
        p_d = 1.*X.sum(axis=1)/X.sum()  # sum(axis=1) -> sum over words
        best_log_likelihood = 0
        
        for m in xrange(self.n_iter):
            if USE_C:
                log_likelihood = em_c.plsa_em(X, p_w_z, p_z_d, p_d)
            else:
                log_likelihood = plsa_em(X, p_w_z, p_z_d, p_d)

            if log_likelihood < best_log_likelihood:
                self.p_w_z, p_w_z = (p_w_z, self.p_w_z)
                self.p_z_d, p_z_d = (p_z_d, self.p_z_d)

    
    def transform(self, X):
        if self.p_w_z is None:
            raise ValueError('Model not fit prior to use')
        p_d = 1.*X.sum(axis=1)/X.sum()
        p_z_d = zeros((self.num_topics, X.shape[0]))
        if USE_C:
            em_c.plsa_em(X, self.p_w_z, p_z_d, p_d, folding=True)
        else:
            plsa_em(X, self.p_w_z, p_z_d, p_d, folding=True)

        return p_z_d.transpose()


    def get_params(self, deep=True):
        """ Eventually, return everything.
            For now, just return a dictionary that
            contains a copy of P(w|z) 
        """ 
        return {'p_w_z': self.p_w_z.copy()}


    def set_params(self, **parameters):
        """ Implement this at some point """
        raise NotImplementedError('set_params needs work...')


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.p_z_d.transpose()


def plsa_em(X, p_w_z, p_z_d, p_d, folding=False,
            min_delta_l=0.0001, max_em_iter=10000):
    """ Expectation Maximization for the PLSA topic model

        Inputs:
          -X:  Input data matrix. (n_samples, n_features)
          -p_w_z:  P(w|z) distribution (array)
          -p_z_d:  P(z|d) distribution (array)
          -p_d:    P(d)   distribution (array)
          -folding:  Transforming new doc-word vectors into topic space
                     definied by P(w|z).  Do NOT update P(w|z)
          -min_delta_l:  Minimum percent change for the log-likelihood.
                         Stop when improvement falls below this threshold.
          -max_em_iter:  Maximum number of iteration for the EM algorithm.
                 
        Returns:
          (float) log-likelihood after final EM iteration.
        
    """
    num_topics = p_z_d.shape[0]
    num_docs, num_words = X.shape
    p_z_wd = zeros((num_topics, num_words, num_docs))

    em_rounds = 0
    """ [None,:] transforms what would be an array 
        with shape (num_topics,) to (1, num_topics)
        so that each word's prob is divided by
        the sum of all words in the topic. 
    """
    if not folding:
        p_w_z[:] = random(p_w_z.shape)
        p_w_z /= p_w_z.sum(axis=0)[None,:]  # sum(axis=0) -> sum over words

    p_z_d[:] = random(p_z_d.shape)
    p_z_d /= p_z_d.sum(axis=0)[None,:]

    l_current = log_likelihood(X, p_w_z, p_z_d, p_d)
    l_old = 0
    em_iter = 0

    while (em_iter < max_em_iter) and \
          (abs(l_current-l_old) > abs(l_old)*min_delta_l):
        # Still improving and fewer than max iterations 
        l_old = l_current
        em_iter += 1

        # Run EM
        em_e_step(p_z_wd, p_w_z, p_z_d)        
        em_m_step(X, p_w_z, p_z_d, p_z_wd, folding)
        l_current = log_likelihood(X, p_w_z, p_z_d, p_d)

    return l_current


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


def em_m_step(X, p_w_z, p_z_d, p_z_wd, folding=False):
    """ Update p_w_z and p_z_d """ 
    num_topics, num_words, num_docs = p_z_wd.shape
    
    # Update p_w_z -- don't touch if folding in.
    if not folding:
        p_w_z[:] = 0.
        for i, j in zip(*X.nonzero()):  # doc, word with nonzero count
            for k in xrange(num_topics):
                p_w_z[j,k] += X[i,j] * p_z_wd[k,j,i]
        
        # Normalize p_w_z
        p_w_z /= p_w_z.sum(axis=0)[None,:]

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
    clf = plsa(n_components=2, n_iter=2)
    clf.fit(X)
