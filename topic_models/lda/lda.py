#!/usr/bin/env python

""" Latent Dirchlet Allocation 

    Uses 'gensim' implementation:
    http://radimrehurek.com/gensim

    This module serves as a wrapper to make
    gensim work more like a scikit-learn
    transformation class. 
""" 

from gensim.matutils import Scipy2Corpus, corpus2csc, corpus2dense
from gensim.models.ldamodel import LdaModel 


class lda:
    """ 
        Inputs:
          -n_components: Number of topics to learn. 
          -n_passes: Number of passes for learning LDA model.
          -update_every: How often to retrain.  Default (0)
                         uses batch mode to learn topics.  
    """ 


    def __init__(self, n_components=2, n_passes=20, update_every=0):
        self.num_topics = n_components 
        self.model = None
        self.n_passes = n_passes
        self.update_every = update_every
    
    
    def fit(self, X, y=None):
        """ Discover a low-dimensional semantic space from X

             Inputs:
               -X: term-document matrix (num_docs, num_words)
               -y: (NOT USED) label vector (num_docs,)
        """ 
        corpus = Scipy2Corpus(X)
        #self.model = LdaModel(corpus, num_topics=self.num_topics,
        #                      passes=self.n_passes, 
        #                      update_every = self.update_every)
        num_words = X.shape[1]
        id_dict = dict(zip(range(num_words), [str(i) for i in xrange(num_words)]))
        self.model = LdaModel(corpus, num_topics=self.num_topics,
                              id2word=id_dict, passes=1, update_every=0)

    
    def transform(self, X):
        """  Create a temporary corpus, cast it into the semantic
             space and pass it back.
        
             Inputs:
               -X: term-document matrix (num_docs, num_words)
        """ 
        num_docs, num_words = X.shape
        temp_corpus = Scipy2Corpus(X)
        temp_corpus = self.model[temp_corpus]

        """ From the gensim docs on corpus2csc: 
            "Convert corpus into a dense numpy 
             array (documents will be columns)"
             hence taking the transpose here: 
        """ 
        return corpus2csc(temp_corpus, num_terms=self.num_topics,
                          num_docs=X.shape[0]).transpose()


    def get_params(self, deep=True):
        if self.model is None:
            raise ValueError('You must fit a model before accessing'
                             ' its parameters!')
        return self.model.__dict__.copy()


    def set_params(self, **parameters):
        """ Implement this at some point """
        raise NotImplementedError('set_params needs work...')


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
